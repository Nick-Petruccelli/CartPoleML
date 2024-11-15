import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, batch_size=32, discount_factor=.95, optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss_fn=keras.losses.MeanSquaredError()):
        input_shape = [4]
        self.n_outputs = 2
        self.model = keras.models.Sequential([
            keras.layers.Dense(32, activation="elu", input_shape=input_shape),
            keras.layers.Dense(32, activation="elu"),
            keras.layers.Dense(self.n_outputs),
        ])
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.discount_factor = discount_factor
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = batch_size

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        else:
            q_values = self.model.predict(state[np.newaxis])
            return np.argmax(q_values[0])

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self, episode):
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.target_model.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = (rewards + (1 - dones) * self.discount_factor * max_next_q_values)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_q_values, q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        if episode % 100 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def train(self, env, n_episodes=600, max_steps=200):
        rewards_per_episode = []
        epsilon_denominator = int(n_episodes * .8)
        for episode in range(n_episodes):
            obs = env.reset()
            reward_total = 0
            for step in range(max_steps):
                epsilon = max(1 - episode / epsilon_denominator, .01)
                obs, reward, done, info = self.play_one_step(env, obs, epsilon)
                reward_total += reward
                if done:
                    rewards_per_episode.append(reward_total)
                    break
            if episode > 50:
                self.training_step(episode)
        plt.plot(rewards_per_episode)
        plt.savefig("plots/fixedTargetDqnRes.png")





if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = DQN()
    model.train(env, n_episodes=1200)

