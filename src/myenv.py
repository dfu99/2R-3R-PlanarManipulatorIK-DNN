import pygame

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot DNN Simulation with Kinematics")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
MAROON = (128, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

COLORS = {'arms': [MAROON, GREEN, BLUE, RED], 'effector': WHITE}

# Robot parameters
ARM1_LENGTH = 100
ARM2_LENGTH = 100
ARM3_LENGTH = 50
ARM4_LENGTH = 50
EFFECTOR_RADIUS = 5