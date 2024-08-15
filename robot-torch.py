import pygame
import torch
from torch import nn
from src.torchnn import RobotNN, DEVICE, test, train, data2loader
from src.myenv import *
from src.testcases import drawCircleAI, drawWavyAI
from robots import Robot3R, Robot2R

if __name__ == "__main__":
    # Make the robot
    thisRobot = Robot3R(400, 300, [ARM1_LENGTH, ARM2_LENGTH, ARM3_LENGTH])
    # thisRobot = Robot2R(400, 300, [ARM1_LENGTH, ARM2_LENGTH])
    # Make the model
    model = RobotNN(3, 3).to(DEVICE)
    # model = RobotNN(2, 2).to(DEVICE)

    # Set the loss function and optimizer
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate random points and convert to PyTorch DataLoader
    data_size = 20000
    # Performance is highly dependent on sufficient training data density
    #   within the expected range of the robot
    dataInput, dataOutput = thisRobot.generate_training(data_size)
    train_dataloader, test_dataloader = data2loader(dataInput, dataOutput, 0.8)

    # Train the model
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    # Main game loop
    running = True
    while running:
        drawCircleAI(thisRobot, (150, 0), model)
        drawWavyAI(thisRobot, model)
        
pygame.quit()