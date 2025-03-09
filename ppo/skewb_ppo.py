import math
from time import sleep
import numpy as np
from stable_baselines3 import PPO
import torch.nn as nn
import datetime

from skewb_multibinary import SkewbEnvB

date_time = datetime.datetime.now()
date = date_time.strftime('%m%d%y')

# modified from https://github.com/WillPalaia/RubiksCubeSolver
# changes made to environment configuration -> integrate with the https://github.com/DoubleGremlin181/RubiksCubeGym environment
# by converting discrete spaces into multibinary
# If this is not done, stable baseline 3 runs EXTREMELY slow

def train_rubiks_cube_solver():
    env = SkewbEnvB()
    obs, _ = env.reset()
    policy_kwargs = dict(activation_fn=nn.ReLU,
                    net_arch=dict(pi=[128, 64], vf=[128, 64]))
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, batch_size = 128, learning_rate = 0.00005)
    print(model.policy)

    training = True
    if training:
        for scrambles in range(1, 15):
            env.scrambles = scrambles
            env.max_moves = scrambles * 2
            print(f"training with {scrambles} scrambles, move limit: {env.max_moves}")
            env.reset()
            model.learn(total_timesteps=int(50000*env.max_moves*2))
        model.save(f"skewb-ppo-shallow-{date}")

    testing = False
    model = model.load(f"skewb-ppo-shallow-{date}")
    if testing:
        for i in range(1,15):
            stats = []
            print("env_scramble: " + str(i))
            moves_i = []
            for j in range(100):
                env.scrambles = i
                env.max_moves = 100
                obs, _ = env.reset()
                
                moves = 0
                done = False
                while not done:
                    action_index, _ = model.predict(obs)
                    obs, rewards, done, _, info = env.step(action_index.tolist())
                    moves += 1

                if done and env.check_solved():
                    moves_i.append(moves)
                    stats.append("1")
                else:
                    stats.append("0")

            
            print(f"Success: {stats.count('1')}/{len(stats)}")
            print(sum(moves_i)/len(moves_i))

if __name__ == "__main__":
    train_rubiks_cube_solver()