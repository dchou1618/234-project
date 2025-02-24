import numpy as np
from rubiks_cube_222_lbl import RubiksCube222EnvLBL
from skewb import SkewbEnv

import pickle
import argparse


def eval_scramble(state):
    number_of_moves = 0
    episode_reward = 0
    success = 0
    done = False

    while not done and number_of_moves <= 100:
        action = np.argmax(q_table[state])
        new_state, reward, done,  _, info = env.step(action)
        episode_reward += reward

        if reward >= 60:
            success = 100

        state = new_state
        number_of_moves += 1

    return episode_reward, success, number_of_moves


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a q_table against random scrambles')
    parser.add_argument('-q', '--q_table', required=True, help="Q Table name")

    args = parser.parse_args()

    #env = RubiksCube222EnvLBL()
    env = SkewbEnv()

    with open(args.q_table, "rb") as f:
        q_table = pickle.load(f)
    
    episode_rewards_dict, episode_success_dict, episode_num_moves_dict = dict(), dict(), dict()

    for i in range(30000):
        if i%1000 == 0:
            episode_rewards_dict[int(i/1000)+1] = []
            episode_success_dict[int(i/1000)+1] = []
            episode_num_moves_dict[int(i/1000)+1] = []
        done = True
        while done:
            #state, _ = env.reset_scramble(num_scramble=int(i/1000)+1)
            state, _ = env.reset(num_scramble=int(i/1000)+1) # for skewb
            _, done = env.reward()

        reward, success, num_moves = eval_scramble(state)
        episode_rewards_dict[int(i/1000)+1].append(reward)
        episode_success_dict[int(i/1000)+1].append(success)
        episode_num_moves_dict[int(i/1000)+1].append(num_moves)
        if i%1000 == 999:
            print("results for shuffle = " + str(int(i/1000)+1))
            print(np.mean(episode_success_dict[int(i/1000)+1]))
            print(np.mean(episode_num_moves_dict[int(i/1000)+1]))
            print("-"*20)
    

   

