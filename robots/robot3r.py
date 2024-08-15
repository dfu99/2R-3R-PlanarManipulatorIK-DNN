from robots.robot import Robot
import numpy as np
from src.utils import pol2cart, cart2pol
from src.myenv import *

try:
    import torch
    from src.torchnn import DEVICE
    NNMODEL = "PyTorch"
except ModuleNotFoundError:
    NNMODEL = "TensorFlow"

class Robot3R(Robot):
    def __init__(self, x, y, arm_lens):
        super().__init__(x, y, arm_lens)

    def IK2D3R(self, x, y, gamma):
        """
        Adapted from
        https://hive.blog/hive-196387/@juecoree/forward-and-reverse-kinematics-for-3r-planar-manipulator
        """
        l1, l2, l3 =  self.arm_lens
        # Position P3
        x3, y3 = np.array([x, y]) - self.origin
        x3 = x3 - (l3 * np.cos(gamma))
        y3 = y3 - (l3 * np.sin(gamma))
        C = np.linalg.norm(np.array([x3, y3]))

        if (l1+l2) > C:
            # angle A and B
            a = np.arccos((l1**2 + l2**2 - C**2)/(2*l1*l2))
            B = np.arccos((l1**2 + C**2 - l2**2)/(2*l1*C))

            # joint angles elbow down
            J1a = np.arctan2(y3, x3) - B
            J2a = np.pi-a
            J3a = gamma - J1a - J2a

            # joint angles elbow up
            J1b = np.arctan2(y3, x3) + B
            J2b = -(np.pi-a)
            J3b = gamma - J1b - J2b

        else:
            print("Warning: Value is unreachable")
            return None
            
        solution1 = (J1a, J2a, J3a)
        solution2 = (J1b, J2b, J3b)
        new_theta = solution1 if solution2 == None else solution2
        # new_theta = solution2 if solution1 == None else solution1
        if not new_theta:
            raise ValueError("No valid solutions")

        self.theta = new_theta
        return new_theta

    def generate_circle_traj(self, n, r, circle_xy):
        self.traj_xy = []
        theta = np.linspace(0, 360, n)
        x0 = self.origin[0] + circle_xy[0]
        y0 = self.origin[1] + circle_xy[1]
        gamma = cart2pol(circle_xy[0], circle_xy[1])[1]
        for t in theta:
            t = np.radians(t)
            x, y = np.array(pol2cart(r, t)) + np.array([x0, y0])
            # gamma value will lock 3rd arm orientation but can also make the trajectory unreachable
            self.traj_xy.append((x, y, np.sum(self.theta)))

    def generate_wavy_traj(self, n):
        self.traj_xy = []
        l1, l2, l3 = self.arm_lens
        theta = np.linspace(0, 360, n)
        r = l1+l2/2
        alim = l1+l2/2
        blim = l1/2
        m = (alim-blim)/2
        wave = np.sin(np.linspace(0, 10*np.pi, n)) * m
        for t, w in zip(theta, wave):
            t = np.radians(t)
            x, y = np.array(pol2cart(r + w, t)) + self.origin
            # gamma value will lock 3rd arm orientation but can also make the trajectory unreachable
            self.traj_xy.append((x, y, np.sum(self.theta)))

    def generate_training(self, n, normalize=True):
        """
        arguments:
        n: number of points
        """
        l1, l2, l3 = self.arm_lens
        self.traj_xy = []
        limits = (l1+l2+l3)

        trainInput = np.zeros((n, 3))
        trainOutput = np.zeros((n, 3))
        gamma = np.random.uniform(0, 2*np.pi)
        for i in range(n):
            while True:
                x, y = self.generate_new_point(limits, limits)
                theta = self.IK2D3R(x, y, gamma)
                if theta != None:
                    break
            self.traj_xy.append((x, y, gamma))
            if normalize:
                trainInput[i, :] = [x - self.origin[0], y - self.origin[1], gamma]
            trainOutput[i, :] = [theta[0], theta[1], theta[2]]
        return (trainInput, trainOutput)

    def generate_new_point(self, xlim, ylim):
        x = np.random.uniform(-xlim, xlim) + self.origin[0]
        y = np.random.uniform(-ylim, ylim) + self.origin[1]
        return x, y

    def moveTo(self, x, y):
        self.traj_xy = []
        lin_vector = np.array([x, y]) - np.array(self.T[2])
        magnitudes = np.linspace(0, 1, 10)
        x0, y0 = self.T[2]
        for rho in magnitudes:
            x1, y1 = np.array([x0, y0]) + rho * lin_vector
            self.traj_xy.append((x1, y1, np.sum(self.theta)))

    def animate(self, model=None):
        for traj in self.traj_xy:
            # Inverse kinematics on the target position to convert it 
            # into angles per each arm
            gamma = traj[2]
            traj = traj[:2]
            if model:
                if NNMODEL == "PyTorch":
                    traj = traj - self.origin
                    test_input = np.concatenate((traj, np.array([gamma])), axis=-1)
                    with torch.no_grad():
                        self.theta = model(torch.FloatTensor(test_input).to(DEVICE)).detach().cpu().numpy()
                elif NNMODEL == "TensorFlow":
                    xtraj, ytraj = traj - self.origin
                    prediction = model.predict(np.array([[xtraj, ytraj, gamma]]))
                    self.theta = prediction.tolist()[0]
            else:
                self.IK2D3R(*traj, gamma)
            # Forward kinematics to find x, y position of each arm
            self.FK(self.theta)
            # Clear the screen
            screen.fill(BLACK)
            # Show the trajectory
            for idx in range(len(self.traj_xy)):
                p1 = self.traj_xy[idx][:2]
                pygame.draw.circle(screen, WHITE, p1, 1)
            # Render
            self.draw(screen)
            # Update the display
            pygame.display.flip()
            # Cap the frame rate
            clock.tick(60)

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()