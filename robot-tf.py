import pygame
from src.myenv import *
from src.testcases import drawCircleAI, drawWavyAI
from robots import Robot3R, Robot2R
from src.tfnn import RobotNN

if __name__ == "__main__":
    # Make the robot
    thisRobot = Robot3R(400, 300, [ARM1_LENGTH, ARM2_LENGTH, ARM3_LENGTH])
    # thisRobot = Robot2R(400, 300, [ARM1_LENGTH, ARM2_LENGTH])
    # Make the model
    model = RobotNN(3, 3)
    # model = RobotNN(2, 2)
    
    # Generate random points and convert to PyTorch DataLoader
    data_size = 20000
    # Performance is highly dependent on sufficient training data density
    #   within the expected range of the robot
    dataInput, dataOutput = thisRobot.generate_training(data_size)

    # Train the model
    model.linear_relu.fit(dataInput, dataOutput, epochs=100, batch_size=64, validation_split=0.2)

    # Main game loop
    running = True
    while running:
        drawCircleAI(thisRobot, (150, 0), model.linear_relu)
        drawWavyAI(thisRobot, model.linear_relu)
        
pygame.quit()