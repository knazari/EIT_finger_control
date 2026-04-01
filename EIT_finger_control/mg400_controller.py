import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

from dobot_api import DobotApi, DobotApiDashboard, DobotApiMove, MyType


class MG400Controller:
    """
    MG400 wrapper with:
    1) Absolute / incremental motion via MovL / MovJ
    2) Stateful jogging via MoveJog
    3) Live feedback reading from port 30004 to track the true TCP pose

    Notes:
    - This version updates self.pose from the robot feedback interface.
    - Software workspace limits are applied to absolute moves and also enforced
      during jog mode using the live TCP pose.
    """

    VALID_JOG_COMMANDS = {
        "X+", "X-",
        "Y+", "Y-",
        "Z+", "Z-",
        "R+", "R-",
        "J1+", "J1-",
        "J2+", "J2-",
        "J3+", "J3-",
        "J4+", "J4-",
        "",
    }

    def __init__(
        self,
        ip: str = "192.168.1.6",
        dashboard_port: int = 29999,
        move_port: int = 30003,
        feed_port: int = 30004,
        speed_factor: int = 8,
        command_pause: float = 0.03,
        min_jog_command_interval: float = 0.05,
        feedback_timeout_s: float = 2.0,
        jog_limit_margin: float = 8.0,
        jog_rotation_limit_margin: float = 6.0,
    ) -> None:
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.feed_port = feed_port
        self.speed_factor = speed_factor
        self.command_pause = command_pause
        self.min_jog_command_interval = min_jog_command_interval
        self.feedback_timeout_s = feedback_timeout_s
        self.jog_limit_margin = jog_limit_margin
        self.jog_rotation_limit_margin = jog_rotation_limit_margin

        self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
        self.move = DobotApiMove(self.ip, self.move_port)
        self.feed = DobotApi(self.ip, self.feed_port)

        self.enabled = False

        # Pose is updated from true robot feedback
        self.pose = {
            "x": 240.0,
            "y": 0.0,
            "z": 120.0,
            "r": 0.0,
        }

        self.limits = {
            "x": (200.0, 300.0),
            "y": (-60.0, 60.0),
            "z": (90.0, 160.0),
            "r": (-90.0, 90.0),
        }

        # Jog state
        self.current_jog_command: str = ""
        self.last_jog_send_time: float = 0.0
        self.jog_reverse_cooldown: float = 0.15
        self.last_jog_change_time: float = 0.0

        # Feedback state
        self.pose_lock = threading.Lock()
        self.feedback_thread: Optional[threading.Thread] = None
        self.feedback_stop_event = threading.Event()
        self.last_feedback_time: float = 0.0
        self.feedback_ok: bool = False
        self.robot_error_state: bool = False
        self.robot_enable_status: Optional[int] = None
        self.algorithm_queue_status = None

    def startup(self) -> None:
        """Clear errors, continue queue if needed, enable robot, set speed, start feedback."""
        print("Starting MG400 controller...")

        try:
            print("ClearError:", self.dashboard.ClearError())
        except Exception as e:
            print(f"ClearError warning: {e}")

        time.sleep(0.3)

        try:
            print("Continue:", self.dashboard.Continue())
        except Exception as e:
            print(f"Continue warning: {e}")

        time.sleep(0.3)

        try:
            print("EnableRobot:", self.dashboard.EnableRobot())
            self.enabled = True
        except Exception as e:
            self.enabled = False
            raise RuntimeError(f"Failed to enable robot: {e}") from e

        time.sleep(1.0)

        try:
            print("SpeedFactor:", self.dashboard.SpeedFactor(self.speed_factor))
        except Exception as e:
            print(f"SpeedFactor warning: {e}")

        self._start_feedback_thread()
        time.sleep(0.3)

        if not self.wait_for_feedback(timeout_s=self.feedback_timeout_s):
            print(
                "Warning: robot feedback did not become ready in time. "
                "Absolute moves may still work, but jog safety limiting will be degraded."
            )

        print("MG400 controller ready.")

    def disable(self) -> None:
        try:
            self.stop_jog()
        except Exception as e:
            print(f"Stop jog before disable warning: {e}")

        self._stop_feedback_thread()

        try:
            print("DisableRobot:", self.dashboard.DisableRobot())
        except Exception as e:
            print(f"DisableRobot warning: {e}")

        self.enabled = False

    def set_speed_factor(self, speed_factor: int) -> None:
        speed_factor = int(max(1, min(100, speed_factor)))
        if speed_factor == self.speed_factor:
            return

        self.speed_factor = speed_factor
        try:
            print("SpeedFactor:", self.dashboard.SpeedFactor(self.speed_factor))
        except Exception as e:
            print(f"SpeedFactor warning: {e}")

    def set_pose_estimate(self, x: float, y: float, z: float, r: float) -> None:
        """
        Kept for compatibility. In this version, pose is normally overwritten by live feedback.
        """
        with self.pose_lock:
            self.pose["x"] = float(x)
            self.pose["y"] = float(y)
            self.pose["z"] = float(z)
            self.pose["r"] = float(r)

    def set_limits(
        self,
        x_lim: tuple,
        y_lim: tuple,
        z_lim: tuple,
        r_lim: tuple,
    ) -> None:
        self.limits["x"] = x_lim
        self.limits["y"] = y_lim
        self.limits["z"] = z_lim
        self.limits["r"] = r_lim

    def get_pose_estimate(self) -> Dict[str, float]:
        with self.pose_lock:
            return dict(self.pose)

    def has_fresh_feedback(self) -> bool:
        return (time.time() - self.last_feedback_time) <= self.feedback_timeout_s

    def wait_for_feedback(self, timeout_s: float = 2.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            if self.feedback_ok and self.has_fresh_feedback():
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _clamp(value: float, limits: tuple) -> float:
        return max(limits[0], min(limits[1], value))

    def _clamp_pose(self, x: float, y: float, z: float, r: float) -> Dict[str, float]:
        return {
            "x": self._clamp(x, self.limits["x"]),
            "y": self._clamp(y, self.limits["y"]),
            "z": self._clamp(z, self.limits["z"]),
            "r": self._clamp(r, self.limits["r"]),
        }

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    def _start_feedback_thread(self) -> None:
        if self.feedback_thread is not None and self.feedback_thread.is_alive():
            return

        self.feedback_stop_event.clear()
        self.feedback_thread = threading.Thread(
            target=self._feedback_loop,
            daemon=True,
        )
        self.feedback_thread.start()

    def _stop_feedback_thread(self) -> None:
        self.feedback_stop_event.set()
        if self.feedback_thread is not None:
            try:
                self.feedback_thread.join(timeout=1.0)
            except Exception:
                pass

    def _feedback_loop(self) -> None:
        """
        Read 1440-byte feedback packets from port 30004 and parse them with MyType.
        This follows the structure used in Dobot's official 4-axis Python demo.
        """
        packet_size = 1440
        has_read = 0

        while not self.feedback_stop_event.is_set():
            try:
                data = bytes()

                while has_read < packet_size and not self.feedback_stop_event.is_set():
                    temp = self.feed.socket_dobot.recv(packet_size - has_read)
                    if len(temp) > 0:
                        has_read += len(temp)
                        data += temp

                if self.feedback_stop_event.is_set():
                    break

                has_read = 0
                feed_info = np.frombuffer(data, dtype=MyType)

                # Official demo validates packets with this magic test_value
                if hex(feed_info["test_value"][0]) != "0x123456789abcdef":
                    continue

                tool_vec = feed_info["tool_vector_actual"][0]

                with self.pose_lock:
                    self.pose["x"] = float(tool_vec[0])
                    self.pose["y"] = float(tool_vec[1])
                    self.pose["z"] = float(tool_vec[2])
                    self.pose["r"] = float(tool_vec[3])

                self.algorithm_queue_status = feed_info["isRunQueuedCmd"][0]
                self.robot_enable_status = int(feed_info["EnableStatus"][0])
                self.robot_error_state = bool(feed_info["ErrorStatus"][0])

                self.feedback_ok = True
                self.last_feedback_time = time.time()

            except Exception as e:
                self.feedback_ok = False
                print(f"[FEEDBACK] Warning: {e}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Absolute / incremental motion helpers
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float, z: float, r: float, linear: bool = True) -> str:
        """
        Move to an absolute target pose after clamping to safe limits.
        Useful for setup and homing.
        """
        if not self.enabled:
            raise RuntimeError("Robot is not enabled.")

        if self.current_jog_command != "":
            self.stop_jog()

        target = self._clamp_pose(x, y, z, r)

        print(
            f"[ABS MOVE] target: "
            f"x={target['x']:.2f}, y={target['y']:.2f}, z={target['z']:.2f}, r={target['r']:.2f}"
        )

        if linear:
            result = self.move.MovL(target["x"], target["y"], target["z"], target["r"])
        else:
            result = self.move.MovJ(target["x"], target["y"], target["z"], target["r"])

        print("Robot response:", result)
        time.sleep(self.command_pause)
        return result

    def move_incremental(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        dr: float = 0.0,
        linear: bool = True,
    ) -> str:
        """
        Move by a small increment relative to the current true feedback pose.
        """
        pose = self.get_pose_estimate()
        target_x = pose["x"] + dx
        target_y = pose["y"] + dy
        target_z = pose["z"] + dz
        target_r = pose["r"] + dr
        return self.move_to(target_x, target_y, target_z, target_r, linear=linear)

    def go_home_estimate(self, linear: bool = False) -> str:
        """
        Return to a known safe setup pose.
        """
        return self.move_to(240.0, 0.0, 120.0, 0.0, linear=linear)

    # ------------------------------------------------------------------
    # Jog motion helpers
    # ------------------------------------------------------------------

    def _validate_jog_command(self, jog_command: str) -> None:
        if jog_command not in self.VALID_JOG_COMMANDS:
            raise ValueError(
                f"Invalid jog command: {jog_command}. "
                f"Expected one of {sorted(self.VALID_JOG_COMMANDS)}"
            )

    @staticmethod
    def _jog_axis_and_sign(jog_command: str) -> Tuple[Optional[str], Optional[str]]:
        if jog_command == "":
            return None, None
        axis = jog_command[0]
        sign = jog_command[1] if len(jog_command) > 1 else None
        return axis, sign

    def _is_opposite_jog(self, new_cmd: str, old_cmd: str) -> bool:
        if new_cmd == "" or old_cmd == "":
            return False

        new_axis, new_sign = self._jog_axis_and_sign(new_cmd)
        old_axis, old_sign = self._jog_axis_and_sign(old_cmd)

        return new_axis == old_axis and new_sign != old_sign

    def _jog_command_allowed_by_limits(self, jog_command: str) -> bool:
        """
        Check live measured TCP pose against software workspace limits before allowing jog.
        """
        if jog_command == "":
            return True

        if not self.has_fresh_feedback():
            print("[JOG LIMIT] Blocking jog because feedback is stale.")
            return False

        pose = self.get_pose_estimate()
        x, y, z, r = pose["x"], pose["y"], pose["z"], pose["r"]

        if jog_command == "X+" and x >= (self.limits["x"][1] - self.jog_limit_margin):
            print(f"[JOG LIMIT] Block X+ at x={x:.2f}")
            return False
        if jog_command == "X-" and x <= (self.limits["x"][0] + self.jog_limit_margin):
            print(f"[JOG LIMIT] Block X- at x={x:.2f}")
            return False

        if jog_command == "Y+" and y >= (self.limits["y"][1] - self.jog_limit_margin):
            print(f"[JOG LIMIT] Block Y+ at y={y:.2f}")
            return False
        if jog_command == "Y-" and y <= (self.limits["y"][0] + self.jog_limit_margin):
            print(f"[JOG LIMIT] Block Y- at y={y:.2f}")
            return False

        if jog_command == "Z+" and z >= (self.limits["z"][1] - self.jog_limit_margin):
            print(f"[JOG LIMIT] Block Z+ at z={z:.2f}")
            return False
        if jog_command == "Z-" and z <= (self.limits["z"][0] + self.jog_limit_margin):
            print(f"[JOG LIMIT] Block Z- at z={z:.2f}")
            return False

        if jog_command == "R+" and r >= (self.limits["r"][1] - self.jog_rotation_limit_margin):
            print(f"[JOG LIMIT] Block R+ at r={r:.2f}")
            return False
        if jog_command == "R-" and r <= (self.limits["r"][0] + self.jog_rotation_limit_margin):
            print(f"[JOG LIMIT] Block R- at r={r:.2f}")
            return False

        return True

    def _send_jog_command(self, jog_command: str, force: bool = False) -> str:
        """
        Low-level MoveJog sender with deduplication and rate limiting.
        """
        if not self.enabled:
            raise RuntimeError("Robot is not enabled.")

        self._validate_jog_command(jog_command)

        now = time.time()

        if (
            not force
            and jog_command == self.current_jog_command
            and (now - self.last_jog_send_time) < self.min_jog_command_interval
        ):
            return f"SKIP_DUPLICATE_JOG({jog_command})"

        result = self.move.MoveJog(jog_command)
        self.current_jog_command = jog_command
        self.last_jog_send_time = now

        print(f"[JOG] MoveJog({repr(jog_command)}) -> {result}")
        return result

    def start_jog(self, jog_command: str) -> str:
        return self._send_jog_command(jog_command, force=False)

    def stop_jog(self) -> str:
        result = self._send_jog_command("", force=True)
        self.last_jog_change_time = time.time()
        return result

    def set_jog_state(self, desired_jog_command: str) -> str:
        """
        Stateful jog controller with:
        - live workspace limiting using true feedback pose
        - duplicate suppression
        - reversal cooldown

        IMPORTANT:
        Limit checking happens before "hold same command" logic so an already
        active jog cannot continue through the soft limit.
        """
        self._validate_jog_command(desired_jog_command)

        # Check limits BEFORE allowing the hold case.
        if desired_jog_command != "" and not self._jog_command_allowed_by_limits(desired_jog_command):
            if self.current_jog_command != "":
                stop_result = self.stop_jog()
                return f"JOG_LIMIT_BLOCKED({desired_jog_command}) | {stop_result}"
            return f"JOG_LIMIT_BLOCKED({desired_jog_command})"

        if desired_jog_command == self.current_jog_command:
            return f"JOG_HOLD({desired_jog_command})"

        now = time.time()

        if self._is_opposite_jog(desired_jog_command, self.current_jog_command):
            if (now - self.last_jog_change_time) < self.jog_reverse_cooldown:
                return f"JOG_REVERSE_BLOCKED({desired_jog_command})"

        if desired_jog_command == "":
            return self.stop_jog()

        result = self.start_jog(desired_jog_command)
        self.last_jog_change_time = time.time()
        return result

    def emergency_stop_like(self) -> Optional[str]:
        """
        Conservative stop helper.
        First stop jog, then try ResetRobot if available.
        """
        try:
            self.stop_jog()
        except Exception as e:
            print(f"Stop jog warning: {e}")

        try:
            result = self.dashboard.ResetRobot()
            print("ResetRobot:", result)
            return result
        except Exception as e:
            print(f"ResetRobot warning: {e}")
            return None