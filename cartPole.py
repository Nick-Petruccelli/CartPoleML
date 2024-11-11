import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras 


def take_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_prob = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_prob)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_prob))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads

def play_multible_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = take_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2,-1,-1):
        discounted[step] += discounted[step+1] * discount_factor
    return discounted

def discount_and_normalize(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(reward, discount_factor) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    mean = flat_rewards.mean()
    std = flat_rewards.std()
    return [(reward-mean)/std for reward in all_discounted_rewards]

def train_model(model, n_iters, n_eps_per_update, n_max_steps, discount_factor, optimizer, loss_fn):
    for iteration in range(n_iters):
        print("Iteration: ", iteration)
        all_rewards, all_grads = play_multible_episodes(env, n_eps_per_update, n_max_steps, model, loss_fn)
        all_final_rewards = discount_and_normalize(all_rewards, discount_factor)
        all_mean_grads = []
        for var_idx in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean([final_reward * all_grads[episode_idx][step][var_idx] for episode_idx, final_rewards in enumerate(all_final_rewards) for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    return model

def run_n_games(n, model):
    for i in range(n):
        obs = env.reset()
        done = False
        while not done:
            left_prob = model(obs[np.newaxis])
            action = (tf.random.uniform([1,1]) > left_prob)
            obs, reward, done, info = env.step(int(action[0,0].numpy()))
            env.render()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_inputs = 4
    model = keras.models.Sequential([
        keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    run_n_games(10, model)

    n_iterations = 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = .95
    optimizer = keras.optimizers.Adam(learning_rate=.01)
    loss_fn = keras.losses.binary_crossentropy
    model = train_model(model, n_iterations, n_episodes_per_update, n_max_steps, discount_factor, optimizer, loss_fn)

    run_n_games(10, model)





