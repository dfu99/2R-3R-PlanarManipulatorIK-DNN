def drawCircle(robot, circle_xy):
    """
    Preset for drawing a circle with the robot arm
    """
    robot.generate_circle_traj(360, 30, circle_xy)
    robot.animate()

def drawCircleAI(robot, circle_xy, model):
    """
    Preset for drawing a circle with the robot arm
    """
    robot.generate_circle_traj(360, 30, circle_xy)
    robot.animate(model=model)

def drawWavyAI(robot, model):
    """
    Preset for drawing a wavy circle around the origin
    """
    robot.generate_wavy_traj(360)
    robot.animate(model=model)