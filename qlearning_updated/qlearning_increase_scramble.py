import numpy as np
import matplotlib.pyplot as plt

import warnings
import pickle
import random
import time

warnings.filterwarnings("ignore")
 
from rubiks_cube_222_lbl import RubiksCube222EnvLBL
from skewb import SkewbEnv
from pyraminx_wo_tips import PyraminxWoTipsEnv

# train function adapted from: https://github.com/DoubleGremlin181/RubiksCubeGym
def train(q_table, env_name, episodes, max_moves = 1000, lr = 0.1, discount = 1, epsilon = 1):

    episode_rewards = []
    episode_success = []
    increment = episodes/14
    decay_rate = 1/(0.9*episodes)

    for episode in range(episodes):
        if episode % 1000 == 1:
            reward_mean = np.mean(episode_rewards[-1000:])
            success_mean = np.mean(episode_success[-1000:])
            print(f"Epsilon:{epsilon:.4f} reward mean: {reward_mean:.4f} "
                      f"success mean: {success_mean:.4f}")
            print()

        # switch depending on environment
        if env_name == "2x2x2":
            state, _ = env.reset_scramble(num_scramble = min(int(episode/increment) + 1, 14))
        elif env_name == "skewb":
            state, _ = env.reset(num_scramble=min(int(episode/increment) + 1, 14))
        else:
            state, _ = env.reset(num_scramble=min(int(episode/increment) + 1, 14))

        episode_reward, success = 0, 0
        done, count = False, 0

        while not done and count <= max_moves: # run for at most maxmoves steps in the environment -> ensure termination
            count += 1 # count of num moves
            if random.uniform(0,1) > epsilon:
                action = np.argmax(q_table[state]) # take the greedy action
            else:
                action = np.random.randint(0, env.action_space.n) # take a random action

            new_state, reward, done, _, info = env.step(action)

            # extimated value of new_state
            val_new_state = np.max(q_table[new_state])
            updated_q = q_table[state][action] + lr * (reward + discount * val_new_state - q_table[state][action])
            q_table[state][action] = updated_q

            episode_reward += reward

            if reward >= 60 and done is True:
                success = 100

            state = new_state

        episode_rewards.append(episode_reward)
        episode_success.append(success)
        epsilon = max(0, epsilon - decay_rate)

    plot(env_name, 1000, episode_rewards, episode_success)


# plotting function obtained from: https://github.com/DoubleGremlin181/RubiksCubeGym
def plot(env_name, GROUP_SIZE, episode_rewards, episode_success):
    moving_avg = np.convolve(episode_rewards, np.ones((GROUP_SIZE,)) / GROUP_SIZE, mode="valid")
    moving_success = np.convolve(episode_success, np.ones((GROUP_SIZE,)) / GROUP_SIZE, mode="valid")

    with open(env_name + "_step_scramble"+op + "_stats" + ".pickle", "wb") as f:
        stats = {"reward": episode_rewards, "success": episode_success}
        pickle.dump(stats, f)

    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(moving_avg))], moving_avg, color='tab:blue', linewidth=2)
    ax.set_ylabel(f"reward {GROUP_SIZE} moving average", color='tab:blue')
    ax.set_xlabel("episode #")

    ax2 = ax.twinx()
    ax2.plot([i for i in range(len(moving_success))], moving_success, color='tab:red', linewidth=2)
    ax2.set_ylabel(f"success {GROUP_SIZE} moving average", color='tab:red')

    fig.savefig(env_name + "_step_scramble"+op + ".png", format='png', dpi=100, bbox_inches='tight')


def close():
    env.close()

def save_qtable(env_name, q_table):
    with open(env_name + "_step_scramble"+op + "_qtable" + ".pickle", "wb") as f:
        pickle.dump(q_table, f)


if __name__ == '__main__':
   
    env = SkewbEnv()

    episodes, max_moves = 280000, 1000
    env_name = "skewb"
    
    if env_name == "2x2x2":
        env = RubiksCube222EnvLBL()
    elif env_name == "skewb":
        env = SkewbEnv()
    else:
        env = PyraminxWoTipsEnv()

    op = f"{env_name}_step_scramble_{episodes}_{max_moves}"
    q_table = np.full([env.observation_space.n, env.action_space.n], 0, dtype=np.float32)

    start = time.time()
    print(start)
    train(q_table, env_name, episodes, max_moves) 
    env.close()
    end = time.time()
    print(end)
    print(end-start)
    save_qtable(env_name, q_table)

