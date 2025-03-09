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

    env = RubiksCube222EnvLBL()
    #env = SkewbEnv()

    with open(args.q_table, "rb") as f:
        q_table = pickle.load(f)
    
    episode_rewards_dict, episode_success_dict, episode_num_moves_dict = [], [], []

    with open("astar.pkl", "rb") as file:
        b = pickle.load(file)
    a = b["states"]

    convert_state = [i.tolist() for i in a]

    for i in range(len(convert_state)):
        env.cube = a[i]
        env.update_cube_reduced()
        env.update_cube_state()
        state = env.cube_state

        reward, success, num_moves = eval_scramble(state)
        episode_rewards_dict.append(reward)
        episode_success_dict.append(success)
        episode_num_moves_dict.append(num_moves)
    

    print(np.mean(episode_success_dict))
    print(np.mean(episode_num_moves_dict))
    print(episode_num_moves_dict)
    print("-"*20)
    

   

