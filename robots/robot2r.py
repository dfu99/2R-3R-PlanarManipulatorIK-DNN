from robots.robot import Robot
import numpy as np
from src.utils import pol2cart, cosine_law
from src.myenv import *
try:
    import torch
    from src.torchnn import DEVICE
    NNMODEL = "PyTorch"
except ModuleNotFoundError:
    NNMODEL = "TensorFlow"


class Robot2R(Robot):
    def __init__(self, x, y, arm_lens):
        super().__init__(x, y, arm_lens)
    
    def IK2D2R(self, x, y):
        """
        From https://www.youtube.com/watch?v=5FD9jyy5eek

        """
        l1, l2 = self.arm_lens
        theta = [0, 0]
        tarx, tary = np.array([x, y]) - self.origin
        C = np.linalg.norm(np.array([tarx, tary]))

        if C > (l1 + l2):
            return None
        cos_theta2 = cosine_law(tarx, tary, l1, l2)
        if abs(cos_theta2) > 1:
            return None
        theta[1] = np.arccos(cos_theta2)
        k1 = l1 + l2 * np.cos(theta[1])
        k2 = l2 * np.sin(theta[1])
        theta[0] = np.arctan2(tary, tarx) - np.arctan2(k2, k1)
        self.theta = (theta[0], theta[1])
        return self.theta
    
    def generate_circle_traj(self, n, r, circle_xy):
        self.traj_xy = []
        theta = np.linspace(0, 360, n)
        # Pick circle origin
        x0 = self.origin[0] + circle_xy[0]
        y0 = self.origin[1] + circle_xy[1]
        for t in theta:
            t = np.radians(t)
            x, y = np.array(pol2cart(r, t)) + np.array([x0, y0])
            self.traj_xy.append((x, y))

    def generate_wavy_traj(self, n):
        self.traj_xy = []
        l1, l2 = self.arm_lens
        theta = np.linspace(0, 360, n)
        r = l1+l2/2
        alim = l1+l2/2
        blim = l1/2
        m = (alim-blim)/2
        wave = np.sin(np.linspace(0, 10*np.pi, n)) * m
        for t, w in zip(theta, wave):
            t = np.radians(t)
            x, y = np.array(pol2cart(r + w, t)) + self.origin
            self.traj_xy.append((x, y))

    def generate_training(self, n, normalize=True):
        """
        arguments:
        n: number of points
        """
        l1, l2 = self.arm_lens
        self.traj_xy = []
        limits = l1+l2
        trainInput = np.zeros((n, 2))
        trainOutput = np.zeros((n, 2))
        for i in range(n):
            while True:
                x, y = self.generate_new_point(limits, limits)
                theta = self.IK2D2R(x, y)
                if theta != None:
                    break
            self.traj_xy.append((x, y))
            if normalize:
                trainInput[i, :] = [x - self.origin[0], y - self.origin[1]]
            trainOutput[i, :] = [theta[0], theta[1]]
        return (trainInput, trainOutput)

    def generate_new_point(self, xlim, ylim):
        x = np.random.uniform(-xlim, xlim) + self.origin[0]
        y = np.random.uniform(-ylim, ylim) + self.origin[1]
        return x, y

    def moveTo(self, x, y):
        self.traj_xy = []
        lin_vector = np.array([x, y]) - np.array(self.T[1])
        magnitudes = np.linspace(0, 1, 10)
        x0, y0 = self.T[1]
        for rho in magnitudes:
            x1, y1 = np.array([x0, y0]) + rho * lin_vector
            self.traj_xy.append((x1, y1))

    def animate(self, model=None):
        for traj in self.traj_xy:
            # Inverse kinematics on the target position to convert it 
            # into angles per each arm
            if model:
                if NNMODEL == "PyTorch":
                    traj = traj - np.array([self.origin[0], self.origin[1]])
                    with torch.no_grad():
                        self.theta = model(torch.FloatTensor(traj).to(DEVICE)).detach().cpu().numpy()
                elif NNMODEL == "TensorFlow":
                    xtraj, ytraj = traj - self.origin
                    prediction = model.predict(np.array([[xtraj, ytraj]]))
                    self.theta = prediction.tolist()[0]
            else:
                self.IK2D2R(*traj)
            # Forward kinematics to find x, y position of each arm
            self.FK(self.theta)
            # Clear the screen
            screen.fill(BLACK)
            # Show the trajectory as a scatter plot
            for idx in range(len(self.traj_xy)):
                p1 = self.traj_xy[idx]
                pygame.draw.circle(screen, WHITE, p1, 1)
            # Render the robot
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