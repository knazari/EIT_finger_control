from mg400_wrapper import MG400
import time

robot = MG400("192.168.1.6")
robot.startup()

robot.move_j(220, 0, 60, 0)
time.sleep(3)
robot.move_l(220, 80, 60, 0)
time.sleep(3)
robot.move_l(220, 0, 60, 0)