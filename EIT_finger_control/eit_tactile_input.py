import copy
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
import serial


class EITTactileInput:
    """
    Reads EIT data, reconstructs the tactile image, and converts it into a compact
    control command for robot teleoperation.

    Output command format:
    {
        "active": bool,
        "mode": "single" or "double",
        "tx": float in [-1, 1],
        "ty": float in [-1, 1],
        "rz": float in [-1, 1],
        "meta": {...}
    }

    Important:
    - get_command() is safe to call from a worker thread.
    - render_latest_plot() must be called from the main thread if plotting is enabled.
    """

    def __init__(
        self,
        serial_port: str,
        baud_rate: int = 115200,
        num_electrodes: int = 16,
        mesh_element_size: float = 0.1,
        sum_abs_threshold: float = 2.0,
        contact_threshold_ratio: float = 0.5,
        min_cluster_size: int = 2,
        angle_opposite_tolerance_deg: float = 45.0,
        enable_plot: bool = True,
        # Temporal double-touch rotation inference
        double_touch_history_len: int = 8,
        double_touch_min_history: int = 4,
        double_touch_motion_threshold_deg: float = 5.0,
        double_touch_rz_value: float = 0.9,
        double_touch_hold_time_s: float = 0.6,
        double_touch_prefer_sign: int = 1,
        # More reliable double-touch detection
        double_touch_min_centroid_separation: float = 0.18,
        double_touch_second_strength_ratio_min: float = 0.20,
        # Robust centroid estimation
        cluster_core_threshold_ratio: float = 0.75,
        centroid_weight_power: float = 2.0,
        centroid_smoothing_alpha: float = 0.35,
        # Manual baseline helper
        raw_frame_buffer_len: int = 10,
    ) -> None:
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.num_electrodes = num_electrodes
        self.mesh_element_size = mesh_element_size
        self.sum_abs_threshold = float(sum_abs_threshold)
        self.contact_threshold_ratio = float(contact_threshold_ratio)
        self.min_cluster_size = int(min_cluster_size)
        self.angle_opposite_tolerance_deg = float(angle_opposite_tolerance_deg)
        self.enable_plot = enable_plot

        # Double-touch temporal inference params
        self.double_touch_history_len = int(double_touch_history_len)
        self.double_touch_min_history = int(double_touch_min_history)
        self.double_touch_motion_threshold_deg = float(double_touch_motion_threshold_deg)
        self.double_touch_rz_value = float(abs(double_touch_rz_value))
        self.double_touch_hold_time_s = float(double_touch_hold_time_s)
        self.double_touch_prefer_sign = 1 if double_touch_prefer_sign >= 0 else -1

        # More reliable double-touch detection params
        self.double_touch_min_centroid_separation = float(double_touch_min_centroid_separation)
        self.double_touch_second_strength_ratio_min = float(double_touch_second_strength_ratio_min)

        # Robust centroid params
        self.cluster_core_threshold_ratio = float(cluster_core_threshold_ratio)
        self.centroid_weight_power = float(centroid_weight_power)
        self.centroid_smoothing_alpha = float(np.clip(centroid_smoothing_alpha, 0.0, 1.0))

        # Manual baseline helper
        self.raw_frame_buffer_len = int(max(1, raw_frame_buffer_len))

        self.sensor = None
        self.v0 = None
        self.v0_initial = None
        self.frame_count = 0

        # Recent raw measurement buffer for manual baseline capture
        self.recent_raw_frames: Deque[np.ndarray] = deque(maxlen=self.raw_frame_buffer_len)
        self.raw_frame_lock = threading.Lock()

        # PyEIT setup
        self.mesh_obj = mesh.create(self.num_electrodes, h0=self.mesh_element_size)
        self.pts = self.mesh_obj.node
        self.tri = self.mesh_obj.element

        self.protocol_obj = protocol.create(
            self.num_electrodes,
            dist_exc=1,
            step_meas=1,
            parser_meas="std",
        )
        self.eit_solver = bp.BP(self.mesh_obj, self.protocol_obj)
        self.eit_solver.setup(weight="none")

        # Plot objects (must only be used from main thread)
        if self.enable_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        else:
            self.fig, self.ax = None, None

        # Thread-safe storage for latest plot-ready data
        self.plot_lock = threading.Lock()
        self.latest_plot_data: Optional[Dict] = None

        # Double-touch temporal state
        self.double_touch_history: Deque[Dict] = deque(maxlen=self.double_touch_history_len)
        self.last_double_touch_rz: float = 0.0
        self.double_touch_hold_until: float = 0.0

    def connect(self) -> None:
        self.sensor = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        print(f"Connected to EIT sensor on {self.serial_port}")
        time.sleep(2.0)

    def close(self) -> None:
        if self.sensor is not None:
            self.sensor.close()
            self.sensor = None

    def read_latest_line(self) -> str:
        """
        Read the latest complete serial line.
        This version is slightly safer and avoids a tight busy loop.
        """
        buffer = b""

        while True:
            if self.sensor is None:
                raise RuntimeError("Serial sensor is not connected.")

            waiting = self.sensor.in_waiting
            if waiting > 0:
                data = self.sensor.read(waiting)
                buffer += data
                lines = buffer.split(b"\n")

                if len(lines) >= 2:
                    latest_line = lines[-2].strip()
                    buffer = lines[-1]
                    if latest_line:
                        return latest_line.decode(errors="ignore")
            else:
                time.sleep(0.001)

    @staticmethod
    def split_eit_data(eit_data_str: str) -> np.ndarray:
        number_strings = [num.strip() for num in eit_data_str.split(",")]
        voltage_data = [float(num) for num in number_strings if num and float(num) != 0]
        return np.array(voltage_data, dtype=float)

    def _store_recent_raw_frame(self, frame: np.ndarray) -> None:
        with self.raw_frame_lock:
            self.recent_raw_frames.append(np.array(frame, copy=True))

    def capture_baseline(self) -> None:
        print("Capturing EIT baseline...")
        self.v0 = None
        self.v0_initial = None

        while self.v0 is None:
            raw = self.read_latest_line()
            arr = self.split_eit_data(raw)
            if len(arr) > 0:
                self.v0 = arr.copy()
                self.v0_initial = arr.copy()
                self._store_recent_raw_frame(arr)

        print("Baseline captured.")

    def capture_current_baseline(self) -> bool:
        """
        Set the baseline to the most recent raw frame.
        Useful for manual recalibration.
        """
        with self.raw_frame_lock:
            if len(self.recent_raw_frames) == 0:
                return False
            new_v0 = np.array(self.recent_raw_frames[-1], copy=True)

        self.v0 = new_v0
        print("Baseline updated from current frame.")
        return True

    def capture_baseline_from_recent_frames(self, num_frames: int = 2) -> bool:
        """
        Set the baseline to the mean of the last num_frames raw frames.
        This is usually more stable than using only one frame.
        """
        n = int(max(1, num_frames))

        with self.raw_frame_lock:
            if len(self.recent_raw_frames) == 0:
                return False

            n = min(n, len(self.recent_raw_frames))
            frames = list(self.recent_raw_frames)[-n:]
            new_v0 = np.mean(np.stack(frames, axis=0), axis=0)

        self.v0 = new_v0
        print(f"Baseline updated from average of last {n} frame(s).")
        return True

    def reconstruct(self, v1: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = float(np.sum(np.abs(self.v0 - v1)))
        if diff <= self.sum_abs_threshold:
            ds = 192.0 * self.eit_solver.solve(self.v0, self.v0, normalize=True)
        else:
            ds = 192.0 * self.eit_solver.solve(v1, self.v0, normalize=True, log_scale=False)
        return diff, ds

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def wrap_angle_deg(angle_deg: float) -> float:
        return (angle_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def point_angle_deg(p: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(p[1], p[0])))

    def clear_double_touch_history(self) -> None:
        self.double_touch_history.clear()
        self.last_double_touch_rz = 0.0
        self.double_touch_hold_until = 0.0

    def _robust_cluster_centroid(self, comp: np.ndarray, ds: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute a more noise-robust centroid for one cluster.

        Strategy:
        - use only the stronger core of the cluster
        - weight strong nodes more heavily
        """
        values = np.clip(ds[comp], a_min=0.0, a_max=None)
        pts = self.pts[comp]

        if len(values) == 0:
            return np.mean(pts, axis=0), 0.0

        local_max = float(np.max(values))
        if local_max <= 1e-12:
            return np.mean(pts, axis=0), 0.0

        core_mask = values >= (self.cluster_core_threshold_ratio * local_max)
        core_pts = pts[core_mask]
        core_vals = values[core_mask]

        if len(core_pts) == 0:
            core_pts = pts
            core_vals = values

        weights = np.power(np.clip(core_vals, 1e-12, None), self.centroid_weight_power)
        weight_sum = float(np.sum(weights))

        if weight_sum <= 1e-12:
            centroid = np.mean(core_pts, axis=0)
            strength = float(np.mean(core_vals))
        else:
            centroid = np.average(core_pts, axis=0, weights=weights)
            strength = float(np.sum(core_vals))

        return centroid, strength

    def find_clusters(self, ds: np.ndarray) -> List[Dict]:
        """
        Finds strong connected regions in node space using a simple graph search
        over mesh nodes. Two nodes are considered neighbors if they appear together
        in a triangle.
        """
        ds = np.asarray(ds).flatten()
        max_val = float(np.max(ds))

        if max_val <= 0:
            return []

        threshold = self.contact_threshold_ratio * max_val
        active_idx = np.where(ds >= threshold)[0]
        active_set = set(active_idx.tolist())

        if len(active_set) == 0:
            return []

        adjacency = {i: set() for i in active_set}
        for tri_nodes in self.tri:
            tri_nodes = list(tri_nodes)
            for i in range(3):
                a = int(tri_nodes[i])
                b = int(tri_nodes[(i + 1) % 3])
                if a in active_set and b in active_set:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        visited = set()
        clusters = []

        for start in active_set:
            if start in visited:
                continue

            stack = [start]
            component = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for nb in adjacency[node]:
                    if nb not in visited:
                        stack.append(nb)

            if len(component) < self.min_cluster_size:
                continue

            comp = np.array(component, dtype=int)
            centroid, strength = self._robust_cluster_centroid(comp, ds)
            angle = self.point_angle_deg(centroid)

            clusters.append(
                {
                    "indices": comp,
                    "centroid": centroid,
                    "strength": strength,
                    "angle_deg": angle,
                }
            )

        clusters.sort(key=lambda c: c["strength"], reverse=True)
        return clusters

    def _assign_consistent_pair(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep cluster ordering consistent across time by matching to the previous pair.
        Uses previously smoothed points for more stable matching.
        """
        if len(self.double_touch_history) == 0:
            return p1, p2

        prev = self.double_touch_history[-1]
        prev_a = prev["p1"]
        prev_b = prev["p2"]

        cost_same = self._distance(p1, prev_a) + self._distance(p2, prev_b)
        cost_swap = self._distance(p2, prev_a) + self._distance(p1, prev_b)

        if cost_swap < cost_same:
            return p2, p1
        return p1, p2

    def _smooth_point(self, new_p: np.ndarray, prev_p: np.ndarray) -> np.ndarray:
        alpha = self.centroid_smoothing_alpha
        return (1.0 - alpha) * prev_p + alpha * new_p

    def _update_double_touch_history(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Store ordered and smoothed centroids.
        """
        p1, p2 = self._assign_consistent_pair(p1, p2)

        if len(self.double_touch_history) > 0:
            prev = self.double_touch_history[-1]
            p1 = self._smooth_point(p1, prev["p1"])
            p2 = self._smooth_point(p2, prev["p2"])

        self.double_touch_history.append(
            {
                "timestamp": time.time(),
                "p1": np.array(p1, copy=True),
                "p2": np.array(p2, copy=True),
                "a1": self.point_angle_deg(p1),
                "a2": self.point_angle_deg(p2),
            }
        )

        return p1, p2

    def _infer_double_touch_rotation(self) -> Tuple[float, Dict]:
        """
        Infer CW/CCW from recent history of the two tracked touch centroids.

        Uses the mean angular motion of the two touch points rather than the
        midpoint angle, which is often unstable for nearly opposite contacts.
        """
        meta = {
            "history_len": len(self.double_touch_history),
            "rotation_source": "centroid_angle_history",
            "angular_delta_deg": 0.0,
            "used_hold": False,
        }

        if len(self.double_touch_history) < self.double_touch_min_history:
            now = time.time()
            if now < self.double_touch_hold_until and abs(self.last_double_touch_rz) > 0.0:
                meta["used_hold"] = True
                return self.last_double_touch_rz, meta
            return 0.0, meta

        hist = list(self.double_touch_history)

        total_delta = 0.0
        for i in range(1, len(hist)):
            da1 = self.wrap_angle_deg(hist[i]["a1"] - hist[i - 1]["a1"])
            da2 = self.wrap_angle_deg(hist[i]["a2"] - hist[i - 1]["a2"])
            pair_delta = 0.5 * (da1 + da2)
            total_delta += pair_delta

        meta["angular_delta_deg"] = total_delta

        if abs(total_delta) >= self.double_touch_motion_threshold_deg:
            rz = self.double_touch_rz_value if total_delta > 0.0 else -self.double_touch_rz_value
            self.last_double_touch_rz = rz
            self.double_touch_hold_until = time.time() + self.double_touch_hold_time_s
            return rz, meta

        now = time.time()
        if now < self.double_touch_hold_until and abs(self.last_double_touch_rz) > 0.0:
            meta["used_hold"] = True
            return self.last_double_touch_rz, meta

        rz = self.double_touch_prefer_sign * self.double_touch_rz_value
        self.last_double_touch_rz = rz
        self.double_touch_hold_until = time.time() + self.double_touch_hold_time_s
        meta["rotation_source"] = "fallback_preferred_sign"
        return rz, meta

    def _is_valid_double_touch_pair(self, c1: Dict, c2: Dict) -> Tuple[bool, Dict]:
        """
        Decide whether the two strongest clusters should be treated as double touch.

        Logic:
        - two separate clusters with enough separation
        - second cluster strong enough relative to the first
        - opposite-ness is kept only as metadata, not as a hard gate
        """
        p1 = c1["centroid"]
        p2 = c2["centroid"]

        sep = self._distance(p1, p2)
        strength1 = float(c1["strength"])
        strength2 = float(c2["strength"])

        if strength1 <= 1e-8:
            strength_ratio = 0.0
        else:
            strength_ratio = strength2 / strength1

        angle_sep = abs(self.wrap_angle_deg(c1["angle_deg"] - c2["angle_deg"]))
        opposite_error = abs(180.0 - angle_sep)

        valid = (
            sep >= self.double_touch_min_centroid_separation
            and strength_ratio >= self.double_touch_second_strength_ratio_min
        )

        meta = {
            "centroid_separation": sep,
            "strength_ratio": strength_ratio,
            "angle_sep_deg": angle_sep,
            "opposite_error_deg": opposite_error,
            "double_touch_valid_pair": valid,
        }
        return valid, meta

    def classify_touch(self, clusters: List[Dict]) -> Dict:
        """
        Returns:
            {
                "mode": "none" | "single" | "double",
                "tx": ...,
                "ty": ...,
                "rz": ...,
                "touch_points": [...],
                "meta": {...}
            }
        """
        if len(clusters) == 0:
            self.clear_double_touch_history()
            return {
                "mode": "none",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "touch_points": [],
                "meta": {},
            }

        if len(clusters) == 1:
            self.clear_double_touch_history()
            c = clusters[0]["centroid"]
            tx = float(np.clip(c[0], -1.0, 1.0))
            ty = float(np.clip(c[1], -1.0, 1.0))
            return {
                "mode": "single",
                "tx": tx,
                "ty": ty,
                "rz": 0.0,
                "touch_points": [c],
                "meta": {},
            }

        # Use the two strongest clusters
        c1 = clusters[0]
        c2 = clusters[1]
        p1 = c1["centroid"]
        p2 = c2["centroid"]

        valid_double, pair_meta = self._is_valid_double_touch_pair(c1, c2)

        if valid_double:
            p1, p2 = self._update_double_touch_history(p1, p2)
            rz, rot_meta = self._infer_double_touch_rotation()

            return {
                "mode": "double",
                "tx": 0.0,
                "ty": 0.0,
                "rz": float(np.clip(rz, -1.0, 1.0)),
                "touch_points": [p1, p2],
                "meta": {
                    **pair_meta,
                    **rot_meta,
                },
            }

        # Fall back to single only if the second cluster is too weak / too close
        self.clear_double_touch_history()
        tx = float(np.clip(p1[0], -1.0, 1.0))
        ty = float(np.clip(p1[1], -1.0, 1.0))
        return {
            "mode": "single",
            "tx": tx,
            "ty": ty,
            "rz": 0.0,
            "touch_points": [p1],
            "meta": pair_meta,
        }

    def update_latest_plot_data(self, ds: np.ndarray, diff: float, command: Dict) -> None:
        """
        Store the latest reconstruction and command for later plotting in the main thread.
        """
        if not self.enable_plot:
            return

        plot_data = {
            "ds": np.array(ds, copy=True),
            "diff": float(diff),
            "command": copy.deepcopy(command),
            "timestamp": time.time(),
        }

        with self.plot_lock:
            self.latest_plot_data = plot_data

    def get_latest_plot_data(self) -> Optional[Dict]:
        """
        Return a copy of the latest plot data.
        """
        if not self.enable_plot:
            return None

        with self.plot_lock:
            if self.latest_plot_data is None:
                return None
            return {
                "ds": np.array(self.latest_plot_data["ds"], copy=True),
                "diff": float(self.latest_plot_data["diff"]),
                "command": copy.deepcopy(self.latest_plot_data["command"]),
                "timestamp": float(self.latest_plot_data["timestamp"]),
            }

    def render_latest_plot(self) -> None:
        """
        Render the latest plot-ready data.
        This must be called from the main thread.
        """
        if not self.enable_plot:
            return

        plot_data = self.get_latest_plot_data()
        if plot_data is None:
            return

        ds = plot_data["ds"]
        diff = plot_data["diff"]
        command = plot_data["command"]

        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])

        self.ax.tripcolor(
            self.pts[:, 0],
            self.pts[:, 1],
            self.tri,
            ds,
            shading="flat",
            cmap="viridis",
        )

        mode = command.get("mode", "none")

        if mode == "single":
            tx = command.get("tx", 0.0)
            ty = command.get("ty", 0.0)

            self.ax.arrow(
                0.0, 0.0, tx * 0.8, ty * 0.8,
                head_width=0.08,
                head_length=0.1,
                fc="red",
                ec="red",
                linewidth=2,
            )
            self.ax.plot(tx, ty, "ro", markersize=8, alpha=0.8)

        elif mode == "double":
            pts = command.get("touch_points", [])
            for p in pts:
                self.ax.plot(p[0], p[1], "ro", markersize=8, alpha=0.8)

            rz = command.get("rz", 0.0)
            label = "ROT +" if rz >= 0 else "ROT -"
            self.ax.text(
                0.0, 1.05, label,
                ha="center",
                va="bottom",
                fontsize=14,
                color="red",
            )

            meta = command.get("meta", {})
            if meta:
                self.ax.text(
                    -1.15,
                    1.05,
                    f"Δθ={meta.get('angular_delta_deg', 0.0):.1f}°",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
                )

        self.ax.set_title(
            f"EIT Teleop | mode={mode} | diff={diff:.2f} | "
            f"tx={command.get('tx', 0.0):.2f} "
            f"ty={command.get('ty', 0.0):.2f} "
            f"rz={command.get('rz', 0.0):.2f}"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def get_command(self) -> Dict:
        """
        Thread-safe tactile acquisition and classification.
        Does NOT call Matplotlib.
        """
        raw = self.read_latest_line()
        v1 = self.split_eit_data(raw)

        if self.v0 is None or self.v0_initial is None:
            raise RuntimeError("Baseline not captured yet.")

        # Keep recent raw frames for manual baseline capture
        self._store_recent_raw_frame(v1)

        if len(v1) != len(self.v0):
            command = {
                "active": False,
                "mode": "single",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "meta": {"reason": "length_mismatch"},
            }
            return command

        self.frame_count += 1

        diff, ds = self.reconstruct(v1)

        # First classify
        if diff <= self.sum_abs_threshold:
            classification = {
                "mode": "none",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "touch_points": [],
                "meta": {},
            }
            self.clear_double_touch_history()
        else:
            clusters = self.find_clusters(ds)
            classification = self.classify_touch(clusters)

        if classification["mode"] == "none":
            plot_command = {
                "mode": "none",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "touch_points": [],
                "meta": {},
            }

            command = {
                "active": False,
                "mode": "single",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "meta": {
                    "diff": diff,
                    "clusters": 0,
                },
            }

            self.update_latest_plot_data(ds, diff, plot_command)
            return command

        command = {
            "active": classification["mode"] in ["single", "double"],
            "mode": classification["mode"],
            "tx": classification["tx"],
            "ty": classification["ty"],
            "rz": classification["rz"],
            "meta": {
                "diff": diff,
                "touch_points": classification["touch_points"],
                **classification.get("meta", {}),
            },
        }

        self.update_latest_plot_data(ds, diff, classification)
        return command