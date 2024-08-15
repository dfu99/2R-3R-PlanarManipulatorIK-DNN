import numpy as np
from src.utils import pol2cart
from src.myenv import *

class Robot:
    """
    Define the Robot
    """
    def __init__(self, x, y, arm_lens):
        self.origin = np.array([x, y])
        self.arm_lens = arm_lens
        self.theta = []
        self.T = self.FK(self.theta)
        self.traj_xy = []
        self.traj_theta = []

    def FK(self, theta):
        """
        Converts the state of the robot from a set of angles in radians (theta1, theta2, theta3)
        To endpoints of each joint in 2D Cartesian (x, y)
        """
        T = [self.origin]
        for i in range(len(theta)):
            xy = T[-1] + np.array(pol2cart(self.arm_lens[i], np.sum(theta[:i+1])))
            T.append(xy)
        self.T = T[1:]
        return self.T

    def draw(self, screen):
        """
        Render the robot on the screen
        """
        arms = np.concatenate((self.origin.reshape((1,2)), self.T))
        for i in range(len(self.T)):
            pygame.draw.line(screen, COLORS['arms'][i], arms[i], arms[i+1], 3)
        # Draw effector
        pygame.draw.circle(screen, WHITE, self.T[-1], EFFECTOR_RADIUS)