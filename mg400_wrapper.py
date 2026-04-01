from dobot_api import DobotApiDashboard, DobotApiMove
import time

class MG400:
    def __init__(self, ip="192.168.1.6"):
        self.dashboard = DobotApiDashboard(ip, 29999)
        self.move = DobotApiMove(ip, 30003)

    def startup(self):
        try:
            self.dashboard.ClearError()
        except:
            pass
        time.sleep(0.5)
        self.dashboard.EnableRobot()
        time.sleep(1)

    def move_j(self, x, y, z, r):
        return self.move.MovJ(x, y, z, r)

    def move_l(self, x, y, z, r):
        return self.move.MovL(x, y, z, r)

    def stop(self):
        try:
            return self.dashboard.DisableRobot()
        except:
            return None