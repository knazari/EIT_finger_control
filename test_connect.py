from dobot_api import DobotApiDashboard, DobotApiMove, DobotApi
import time

ROBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003
FEED_PORT = 30004

def main():
    print("Connecting to MG400...")
    dashboard = DobotApiDashboard(ROBOT_IP, DASHBOARD_PORT)
    move = DobotApiMove(ROBOT_IP, MOVE_PORT)
    feed = DobotApi(ROBOT_IP, FEED_PORT)
    print("Connected.")

    # Enable robot
    print("Enabling robot...")
    print(dashboard.EnableRobot())
    time.sleep(2)

    # Optional: clear errors if needed
    # print(dashboard.ClearError())

    # Move to a safe test pose
    # IMPORTANT: adjust to a safe pose for your setup
    print("Sending MovJ...")
    print(move.MovJ(200, 0, 50, 0))

    time.sleep(5)

    print("Done.")
    # Depending on SDK implementation, sockets may close on process exit

if __name__ == "__main__":
    main()