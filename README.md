# 234 Project

We are interested in examining the performance of various reinforcement learning algorithms in the derivation of optimal policies for 3 distinct (and likely, less common) variants of Rubik’s Cube - namely the 2x2x2, Pyraminx and Skewb. This is an interesting problem given the relatively large state spaces (Skewb: ~100k, 2x2x2: ~3.6 million, Pyraminx: ~76 million) and - in the case of international competitions - the emphasis on solving the cube in the least amount of time/ number of steps. Fundamentally, this is a reinforcement learning problem wherein our objective is to find the shortest path from any arbitrary state of the Rubik’s cube to the singular end state (solved cube). Additionally, given that there is only a singular end state, this means that feedback from the environment in terms of reward will be exceptionally sparse, which makes this problem more difficult.

We will be using the third party RubiksCubeGym environment as the simulator, with the possibility of making changes to the reward function to better capture penalties for taking too many steps.

The creators of RubiksCubeGym predominantly explored how parallel Q-learning performed across the 3 variants. In addition to parallel Q-learning, we will explore PPO, Curriculum Learning (CL)/AutoCL, and DeepCubeA.

Existing work on using reinforcement learning to solve Rubik’s cube has been focused on the “standard” 3x3x3 cube, with less work done on whether similar methods will fare as well in other variants of the Rubik’s cube. A study that we intend to survey in greater detail would be Agostinelli et al [1]. They proposed the usage of DeepCubeA (a modification of DeepCube) to solve the 3x3x3 Rubik’s cube and other logic puzzles. Particularly, the authors first used a deep neural network to approximate the optimal cost-to-go function (cost of taking shortest path to end state) which is then used as the heuristic for A* search before running DeepCube. Not only was DeepCubeA able to find the solution from all test states, it was able to find the shortest path 60% of the time. It would hence be interesting to see if DeepCubeA can be extended to solve other variants of Rubiks.
Another area that we intend to explore would be that of curriculum learning, which is inspired by [2]. Curriculum learning is a technique wherein our model will be trained on examples with increasing difficulty. In the context of Rubik’s Cube, increasing the number of times we randomly shuffle from the solved state could constitute as varying the difficulty. The model is only exposed to harder scrambles once it successfully solves lower difficulty scrambles earlier in its learning trajectory. Before delving deeper into this, we would likely be reading up on more literature with regards to this technique.

We will be evaluating the various algorithms based on solve rate for each of the 3 Rubik cube variants, as well as the average number of steps taken to reach the solution from a random starting state. We will plot the cumulative reward as well as the solve rate (number of iterations until solved) against the number of training episodes to compare convergence rates between the three learning methods. As part of sensitivity analysis, over different learning rates, discount rates, and batch sizes, we will plot solve rates of each of the cubes over 100, 200, 300 training iterations. 

References
[1] Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and search." Nature Machine Intelligence 1.8 (2019): 356-363.
[2] Aitsaid, Azzedine. “(Automatic)Curriculum Learning : Solving Rubik’s Cube Beyond PPO Limitations.” Medium, 23 Oct. 2023, medium.com/@ja_aitsaid/automatic-curriculum-learning-solving-rubiks-cube-beyond-ppo-limitations-3e6489b2ff6f. 

## Q-learning

## PPO

## DeepCubeA

1. Setup with python venv and `pip install -r requirements.txt`.
2. Export the appropriate python paths so relative imports are recognized. See `example.sh`. One would run `source ./example.sh`
3. `python ctg_approx/avi.py --env cube222 --states_per_update 500 --batch_size 100  --nnet_name cube222 --max_itrs 1000000 --loss_thresh 0.1 --back_max 1000 --num_update_procs 8`
