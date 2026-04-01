import time

from mg400_controller import MG400Controller
from tactile_teleop_core import TactileTeleopCore


def print_pose(robot: MG400Controller) -> None:
    pose = robot.get_pose_estimate()
    print(
        f"Current pose estimate -> "
        f"x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}, r={pose['r']:.2f}"
    )


def run_dummy_sequence(robot: MG400Controller, teleop: TactileTeleopCore) -> None:
    """
    A short sequence of fake tactile commands.
    These are deliberately small and slow.
    """

    dummy_commands = [
        # Single-touch tests
        {"name": "single +X", "cmd": {"active": True, "mode": "single", "tx": 0.7, "ty": 0.0, "rz": 0.0}},
        {"name": "single -X", "cmd": {"active": True, "mode": "single", "tx": -0.7, "ty": 0.0, "rz": 0.0}},
        {"name": "single +Y", "cmd": {"active": True, "mode": "single", "tx": 0.0, "ty": 0.7, "rz": 0.0}},
        {"name": "single -Y", "cmd": {"active": True, "mode": "single", "tx": 0.0, "ty": -0.7, "rz": 0.0}},

        # Double-touch / rotation tests
        {"name": "double +R", "cmd": {"active": True, "mode": "double", "tx": 0.0, "ty": 0.0, "rz": 0.7}},
        {"name": "double -R", "cmd": {"active": True, "mode": "double", "tx": 0.0, "ty": 0.0, "rz": -0.7}},

        # No-touch test
        {"name": "inactive", "cmd": {"active": False, "mode": "single", "tx": 0.0, "ty": 0.0, "rz": 0.0}},
    ]

    for item in dummy_commands:
        print("\n" + "=" * 60)
        print(f"Executing command: {item['name']}")
        print(f"Input tactile command: {item['cmd']}")

        dx, dy, dz, dr = teleop.compute_increment(item["cmd"])
        print(f"Computed increment -> dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, dr={dr:.2f}")

        if dx == 0.0 and dy == 0.0 and dz == 0.0 and dr == 0.0:
            print("No robot motion for this command.")
        else:
            robot.move_incremental(dx=dx, dy=dy, dz=dz, dr=dr, linear=True)

        print_pose(robot)
        time.sleep(1.2)


def run_manual_loop(robot: MG400Controller, teleop: TactileTeleopCore) -> None:
    """
    Simple keyboard-based dummy command loop.
    This is useful for fast testing before connecting the EIT sensor.
    """
    print("\nEntering manual test loop.")
    print("Controls:")
    print("  w/s -> +Y / -Y")
    print("  d/a -> +X / -X")
    print("  e/q -> +R / -R")
    print("  x   -> no motion")
    print("  h   -> go to estimated home")
    print("  p   -> print pose")
    print("  z   -> exit")

    mapping = {
        "w": {"active": True, "mode": "single", "tx": 0.0, "ty": 0.8, "rz": 0.0},
        "s": {"active": True, "mode": "single", "tx": 0.0, "ty": -0.8, "rz": 0.0},
        "d": {"active": True, "mode": "single", "tx": 0.8, "ty": 0.0, "rz": 0.0},
        "a": {"active": True, "mode": "single", "tx": -0.8, "ty": 0.0, "rz": 0.0},
        "e": {"active": True, "mode": "double", "tx": 0.0, "ty": 0.0, "rz": 0.8},
        "q": {"active": True, "mode": "double", "tx": 0.0, "ty": 0.0, "rz": -0.8},
        "x": {"active": False, "mode": "single", "tx": 0.0, "ty": 0.0, "rz": 0.0},
    }

    while True:
        key = input("\nCommand key: ").strip().lower()

        if key == "z":
            print("Exiting manual loop.")
            break

        if key == "h":
            print("Going to estimated home pose...")
            robot.go_home_estimate(linear=False)
            print_pose(robot)
            continue

        if key == "p":
            print_pose(robot)
            continue

        if key not in mapping:
            print("Unknown key.")
            continue

        tactile_cmd = mapping[key]
        print(f"Tactile cmd: {tactile_cmd}")

        dx, dy, dz, dr = teleop.compute_increment(tactile_cmd)
        print(f"Computed increment -> dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, dr={dr:.2f}")

        if dx == 0.0 and dy == 0.0 and dz == 0.0 and dr == 0.0:
            print("No robot motion for this command.")
            continue

        robot.move_incremental(dx=dx, dy=dy, dz=dz, dr=dr, linear=True)
        print_pose(robot)


def main() -> None:
    robot = MG400Controller(
        ip="192.168.1.6",
        speed_factor=10,
        command_pause=0.2,
    )

    teleop = TactileTeleopCore(
        translation_gain_mm=4.0,
        rotation_gain_deg=4.0,
        deadband=0.15,
        invert_x=False,
        invert_y=False,
        invert_rz=False,
    )

    robot.startup()

    # IMPORTANT:
    # Set this to the actual robot pose where you start the test.
    # For example, first move the robot manually to a safe central pose,
    # then set that same pose here.
    robot.set_pose_estimate(220.0, 0.0, 120.0, 0.0)

    # Conservative limits for early testing
    robot.set_limits(
        x_lim=(190.0, 250.0),
        y_lim=(-60.0, 60.0),
        z_lim=(100.0, 140.0),
        r_lim=(-45.0, 45.0),
    )

    print_pose(robot)

    print("\nRunning automatic dummy sequence first...")
    run_dummy_sequence(robot, teleop)

    print("\nNow you can test manually.")
    run_manual_loop(robot, teleop)

    print("\nDisabling robot...")
    robot.disable()
    print("Done.")


if __name__ == "__main__":
    main()