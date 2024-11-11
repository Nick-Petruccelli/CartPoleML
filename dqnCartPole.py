import gym
import tensorflow as tf
from tensorflow import keras
import numpy

def make_model():
    input_shape = [4]
    n_outputs = 2
    model = keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=input_shape),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(n_outputs),
    ])
    return model



if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = make_model()

