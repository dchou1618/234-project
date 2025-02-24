import numpy as np
import matplotlib.pyplot as plt

import warnings
from datetime import datetime
import argparse
import pickle

warnings.filterwarnings("ignore")
 
from rubiks_cube_222_lbl import RubiksCube222EnvLBL
from skewb import SkewbEnv
from pyraminx_wo_tips import PyraminxWoTipsEnv

# modified from: https://github.com/DoubleGremlin181/RubiksCubeGym

def train(q_table):
    EPISODES = episodes_per_process
    GROUP_SIZE = 1000
    LEARNING_RATE = 0.1
    DISCOUNT = 1
    EPSILON = 1.0
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES * 0.9
    EPSILON_DECAY = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    episode_rewards = []
    episode_success = []

    for episode in range(EPISODES):
        if episode % GROUP_SIZE == 1:
            print(f"Epsilon:{EPSILON:.4f} {GROUP_SIZE} reward mean: {np.mean(episode_rewards[-GROUP_SIZE:]):.4f} "
                      f"{GROUP_SIZE} success mean: {np.mean(episode_success[-GROUP_SIZE:]):.4f}")
            print()

        #state, _ = env.reset_scramble(num_scramble=min(int(episode/10000) + 1, 14))
        #for skewb
        state, _ = env.reset(num_scramble=min(int(episode/10000) + 1, 14))

        episode_reward = 0
        success = 0
        done = False
        count = 0

        while not done and count <= 1000: # run for at most 1000 steps in the environment -> ensure termination
            count += 1
            if np.random.randn() > EPSILON:
                action = np.argmax(q_table[state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _, info = env.step(action)

            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state][action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state][action] = new_q

            episode_reward += reward

            if reward >= 60:
                success = 100

            state = new_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            EPSILON = max(EPSILON - EPSILON_DECAY, 0)

        episode_rewards.append(episode_reward)
        episode_success.append(success)

    plot(GROUP_SIZE, episode_rewards, episode_success)


def plot(GROUP_SIZE, episode_rewards, episode_success):
    moving_avg = np.convolve(episode_rewards, np.ones((GROUP_SIZE,)) / GROUP_SIZE, mode="valid")
    moving_success = np.convolve(episode_success, np.ones((GROUP_SIZE,)) / GROUP_SIZE, mode="valid")

    with open("skewb_step_scramble"+op + "_stats" + ".pickle", "wb") as f:
        stats = {"reward": episode_rewards, "success": episode_success}
        pickle.dump(stats, f)

    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(moving_avg))], moving_avg, color='tab:blue', linewidth=2)
    ax.set_ylabel(f"reward {GROUP_SIZE} moving average", color='tab:blue')
    ax.set_xlabel("episode #")

    ax2 = ax.twinx()
    ax2.plot([i for i in range(len(moving_success))], moving_success, color='tab:red', linewidth=2)
    ax2.set_ylabel(f"success {GROUP_SIZE} moving average", color='tab:red')

    fig.savefig("skewb_step_scramble"+op + ".png", format='png', dpi=100, bbox_inches='tight')


def close():
    env.close()

def save_qtable(q_table):
    with open("skewb_step_scramble"+op + "_qtable" + ".pickle", "wb") as f:
        pickle.dump(q_table, f)


if __name__ == '__main__':
    #env = RubiksCube222EnvLBL()
    env = SkewbEnv()

    parser = argparse.ArgumentParser(description='q learning increase scramble')
    parser.add_argument('-s', '--size', type=int, default=280000, help="Number of episodes")
    args = parser.parse_args()
    episodes_per_process = args.size
    
    op = f"skewb_step_scramble_{args.size}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    t = datetime.now()

    q_table = np.full([env.observation_space.n, env.action_space.n], 0, dtype=np.float32)

    train(q_table)

    env.close()
    save_qtable(q_table)

