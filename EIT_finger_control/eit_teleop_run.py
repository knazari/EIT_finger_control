import copy
import threading
import time
from typing import Dict

from mg400_controller import MG400Controller
from tactile_teleop_core import TactileTeleopCore
from eit_tactile_input import EITTactileInput


class SharedTactileState:
    """
    Thread-safe container for the latest tactile command.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cmd: Dict = {
            "active": False,
            "mode": "single",
            "tx": 0.0,
            "ty": 0.0,
            "rz": 0.0,
            "timestamp": 0.0,
            "meta": {},
        }

    def set(self, cmd: Dict) -> None:
        with self.lock:
            self.cmd = copy.deepcopy(cmd)
            self.cmd["timestamp"] = time.time()

    def get(self) -> Dict:
        with self.lock:
            return copy.deepcopy(self.cmd)

    def clear(self) -> None:
        with self.lock:
            self.cmd = {
                "active": False,
                "mode": "single",
                "tx": 0.0,
                "ty": 0.0,
                "rz": 0.0,
                "timestamp": time.time(),
                "meta": {"reason": "manual_clear"},
            }


class SharedManualCommandState:
    """
    Thread-safe container for manual commands such as baseline reset or homing.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.request_baseline_refresh = False
        self.request_home_move = False

    def request_refresh(self) -> None:
        with self.lock:
            self.request_baseline_refresh = True

    def request_home(self) -> None:
        with self.lock:
            self.request_home_move = True

    def consume_refresh_request(self) -> bool:
        with self.lock:
            if self.request_baseline_refresh:
                self.request_baseline_refresh = False
                return True
            return False

    def consume_home_request(self) -> bool:
        with self.lock:
            if self.request_home_move:
                self.request_home_move = False
                return True
            return False


def tactile_reader_loop(
    eit: EITTactileInput,
    shared_state: SharedTactileState,
    stop_event: threading.Event,
    print_rate_hz: float = 1.0,
) -> None:
    """
    Continuously reads tactile data and updates the shared latest command.
    This thread does NOT do any plotting.
    """
    print_period = 1.0 / print_rate_hz if print_rate_hz > 0 else 0.0
    last_print_time = 0.0

    while not stop_event.is_set():
        try:
            tactile_cmd = eit.get_command()
            shared_state.set(tactile_cmd)

            now = time.time()
            if print_period > 0 and (now - last_print_time) >= print_period:
                print("TACTILE CMD:", tactile_cmd)
                last_print_time = now

        except Exception as e:
            print(f"[TACTILE THREAD] Warning: {e}")
            time.sleep(0.05)


def keyboard_input_loop(
    manual_state: SharedManualCommandState,
    stop_event: threading.Event,
) -> None:
    """
    Wait for keyboard input in a background thread.

    Commands:
    - Enter          -> manually refresh EIT baseline from recent raw frames
    - h + Enter      -> move robot to home pose
    - q + Enter      -> quit
    """
    print("Keyboard commands: Enter=baseline refresh | h=home | q=quit")

    while not stop_event.is_set():
        try:
            user_text = input()

            if stop_event.is_set():
                break

            cmd = user_text.strip().lower()

            if cmd == "q":
                print("Quit requested from keyboard.")
                stop_event.set()
                break

            if cmd == "":
                manual_state.request_refresh()
                print("Manual baseline refresh requested.")
            elif cmd == "h":
                manual_state.request_home()
                print("Manual home move requested.")
            else:
                print("Unknown command. Use Enter=baseline refresh | h=home | q=quit")

        except EOFError:
            break
        except Exception as e:
            print(f"[KEYBOARD THREAD] Warning: {e}")
            time.sleep(0.1)


def main():
    # -------------------------------
    # Robot
    # -------------------------------
    robot = MG400Controller(
        ip="192.168.1.6",
        speed_factor=10,
        command_pause=0.03,
        min_jog_command_interval=0.05,
    )
    robot.startup()

    print("Moving robot to home pose...")
    robot.go_home_estimate(linear=False)
    time.sleep(1.0)

    enable_robot_control = True

    # Optional pose estimate for setup/homing utilities only
    robot.set_pose_estimate(220.0, 0.0, 120.0, 0.0)

    # These limits are only used by MovL/MovJ helper functions,
    # not by MoveJog directly
    robot.set_limits(
        x_lim=(200.0, 300.0),
        y_lim=(-90.0, 90.0),
        z_lim=(100.0, 140.0),
        r_lim=(-90.0, 90.0),
    )

    # -------------------------------
    # Teleop mapping (JOG-based)
    # -------------------------------
    teleop = TactileTeleopCore(
        deadband=0.10,
        filter_alpha=0.45,
        invert_x=False,
        invert_y=True,
        invert_rz=False,
        normalize_translation=True,
        dominant_axis_ratio=1.15,
        axis_switch_ratio=1.35,
        min_speed_factor=10,
        max_speed_factor=35,
        rotation_min_speed_factor=10,
        rotation_max_speed_factor=30,
        sensor_angle_offset_deg=-90.0, # adjust for wligning rotation ...
        rotation_hold_frames=4,
    )

    # -------------------------------
    # EIT input
    # -------------------------------
    eit = EITTactileInput(
        serial_port="COM5",
        baud_rate=115200,
        num_electrodes=16,
        mesh_element_size=0.1,
        sum_abs_threshold=5.0,
        contact_threshold_ratio=0.55,
        min_cluster_size=8,
        angle_opposite_tolerance_deg=60.0,
        enable_plot=True,
        double_touch_history_len=5,
        double_touch_min_history=4,
        double_touch_motion_threshold_deg=7.0,
        double_touch_rz_value=1.1,
        double_touch_hold_time_s=0.9,
        double_touch_prefer_sign=1,
        double_touch_min_centroid_separation=0.05,
        double_touch_second_strength_ratio_min=0.10,
        raw_frame_buffer_len=10,
    )

    eit.connect()
    eit.capture_baseline()

    # -------------------------------
    # Shared thread state
    # -------------------------------
    shared_state = SharedTactileState()
    manual_state = SharedManualCommandState()
    stop_event = threading.Event()

    tactile_thread = threading.Thread(
        target=tactile_reader_loop,
        args=(eit, shared_state, stop_event),
        daemon=True,
    )
    tactile_thread.start()

    keyboard_thread = threading.Thread(
        target=keyboard_input_loop,
        args=(manual_state, stop_event),
        daemon=True,
    )
    keyboard_thread.start()

    # -------------------------------
    # Loop settings
    # -------------------------------
    control_rate_hz = 30.0

    # If tactile command is too old, stop jog
    stale_timeout_s = 0.20

    # Plot update throttling
    plot_rate_hz = 15.0
    plot_period = 1.0 / plot_rate_hz
    last_plot_time = 0.0

    # Debug print throttling for jog state
    jog_print_rate_hz = 0.5
    jog_print_period = 1.0 / jog_print_rate_hz
    last_jog_print_time = 0.0

    print("Starting EIT jog teleoperation loop. Press Ctrl+C to stop.")

    try:
        next_time = time.time()

        while not stop_event.is_set():
            now = time.time()

            # ----------------------------------
            # Manual baseline refresh on Enter
            # ----------------------------------
            if manual_state.consume_refresh_request():
                try:
                    success = eit.capture_baseline_from_recent_frames(num_frames=2)
                    if success:
                        print("EIT baseline manually refreshed from average of last 2 frames.")
                    else:
                        print("Baseline refresh failed: not enough recent raw frames yet.")
                except Exception as e:
                    print(f"Manual baseline refresh failed: {e}")

            # ----------------------------------
            # Manual move to home on 'h'
            # ----------------------------------
            if manual_state.consume_home_request():
                try:
                    print("Stopping jog and moving robot to home pose...")
                    robot.stop_jog()
                    time.sleep(0.1)
                    robot.go_home_estimate(linear=False)
                    shared_state.clear()
                    print("Robot moved to home pose.")
                except Exception as e:
                    print(f"Manual home move failed: {e}")

            tactile_cmd = shared_state.get()

            cmd_age = now - tactile_cmd.get("timestamp", 0.0)
            stale = cmd_age > stale_timeout_s

            if stale:
                tactile_cmd["active"] = False

            jog_state = teleop.compute_jog_command(tactile_cmd)

            if stale:
                jog_state["jog_command"] = ""
                jog_state["active"] = False

            if enable_robot_control:
                # Update speed first
                robot.set_speed_factor(jog_state["speed_factor"])

                # Then set desired jog state
                robot.set_jog_state(jog_state["jog_command"])

            # Periodic jog debug printing
            if (now - last_jog_print_time) >= jog_print_period:
                print("JOG STATE:", jog_state)
                last_jog_print_time = now

            # Plot update in main thread only
            if eit.enable_plot and (now - last_plot_time) >= plot_period:
                try:
                    eit.render_latest_plot()
                except Exception as e:
                    print(f"[PLOT] Warning: {e}")
                last_plot_time = now

            # Fixed-rate loop timing
            next_time += 1.0 / control_rate_hz
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()

    except KeyboardInterrupt:
        print("Stopping teleoperation...")

    finally:
        stop_event.set()

        try:
            tactile_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            keyboard_thread.join(timeout=0.5)
        except Exception:
            pass

        try:
            robot.stop_jog()
        except Exception as e:
            print(f"Stop jog warning: {e}")

        try:
            eit.close()
        except Exception as e:
            print(f"EIT close warning: {e}")

        try:
            robot.disable()
        except Exception as e:
            print(f"Robot disable warning: {e}")

        print("Shutdown complete.")


if __name__ == "__main__":
    main()