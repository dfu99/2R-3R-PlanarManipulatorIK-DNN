import tensorflow as tf
from tensorflow.keras import layers, models

class RobotNN:
    def __init__(self, inputs, outputs):
        self.linear_relu = models.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(inputs,)),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(outputs)
                ])
        self.linear_relu.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])