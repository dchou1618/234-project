# 234 Project

We are interested in examining the performance of various reinforcement learning algorithms in the derivation of optimal policies for 3 distinct (and likely, less common) variants of Rubik’s Cube - namely the 2x2x2, Pyraminx and Skewb. This is an interesting problem given the relatively large state spaces (Skewb, 2x2x2: ~3.6 million, Pyraminx: ~900k) and - in the case of international competitions - the emphasis on solving the cube in the least amount of time/ number of steps. Fundamentally, this is a reinforcement learning problem wherein our objective is to find the shortest path from any arbitrary state of the Rubik’s cube to the singular end state (solved cube). Additionally, given that there is only a singular end state, this means that feedback from the environment in terms of reward will be exceptionally sparse, which makes this problem more difficult.

We will be using the third party RubiksCubeGym environment as the simulator, with the possibility of making changes to the reward function to better capture penalties for taking too many steps.

We will explore PPO (reward augmentation, policy param), Q-learning, Curriculum Learning (CL), and DeepCubeA.

Existing work on using reinforcement learning to solve Rubik’s cube has been focused on the “standard” 3x3x3 cube, with less work done on whether similar methods will fare as well in other variants of the Rubik’s cube. A study that we intend to survey in greater detail would be Agostinelli et al [1]. They proposed the usage of DeepCubeA (a modification of DeepCube) to solve the 3x3x3 Rubik’s cube and other logic puzzles. Particularly, the authors first used a deep neural network to approximate the optimal cost-to-go function (cost of taking shortest path to end state) which is then used as the heuristic for A* search before running DeepCube. Not only was DeepCubeA able to find the solution from all test states, it was able to find the shortest path 60% of the time. It would hence be interesting to see if DeepCubeA can be extended to solve other variants of Rubiks.

Another area that we intend to explore would be that of curriculum learning, which is inspired by [2]. Curriculum learning is a technique wherein our model will be trained on examples with increasing difficulty. In the context of Rubik’s Cube, increasing the number of times we randomly shuffle from the solved state could constitute as varying the difficulty. The model is only exposed to harder scrambles once it successfully solves lower difficulty scrambles earlier in its learning trajectory. Before delving deeper into this, we would likely be reading up on more literature with regards to this technique.

## Q-learning & PPO
1. Setup with conda venv and `conda env create -f environment.yml`
2. Follow instructions in the README in the respective folders to run the code
Trained model .zip files and data used to plot the various fig can be found here: https://drive.google.com/drive/folders/1Djy6SVJ3gy8FSlgjCcoVjtJluBlRwd8G?usp=share_link

## DeepCubeA

We added environments for 2x2, Skewb, and Pyraminx under `DeepCubeA/environments`. Item 3 below is used to train and save the cost-to-go heuristic neural networks. `generate_astar_data.py` generates scrambled states used as input for the search algorithms bfs and greedy best-first search (gbfs) that reuse the saved cost-to-go neural networks. The solution paths of gbfs along with the number of scrambles, runtime, and number of states explored are stored in a pickle file under `data/{env}/train/data_0.pkl`. After running 4 listed below, we run `generate_plots.py` to generate the plots to compare gbfs' solution in `data/{env}/train/data_0.pkl` and astar's solution under `results/{env}/results.pkl`. The outputs and saved models are in the [drive link](https://drive.google.com/drive/folders/1qhsDOiprHPDzmaPAvM0Onwlh_7hlNRKZ?usp=sharing)

1. Setup with python venv and `pip install -r requirements.txt`.
2. Export the appropriate python paths so relative imports are recognized. See `example.sh`.
3. `python ctg_approx/avi.py --env cube222 --states_per_update 500000 --batch_size 5000  --nnet_name cube222 --max_itrs 10000 --loss_thresh 0.01 --back_max 14 --num_update_procs 4 --lr 5e-4 --lr_d 0.99999`. The number of states per update needs to exceed the batch size - otherwise, the batch data will be empty.
4. `python search_methods/astar.py --states data/cube222/train/data_0.pkl --model saved_models/cube222/current/ --env cube222 --weight 0.8 --batch_size 5000 --results_dir results/cube222/ --language python --nnet_batch_size 5000` for astar

## References

[1] Agostinelli, Forest, et al. "Solving the Rubik’s cube with deep reinforcement learning and search." Nature Machine Intelligence 1.8 (2019): 356-363.

[2] Aitsaid, Azzedine. “(Automatic)Curriculum Learning : Solving Rubik’s Cube Beyond PPO Limitations.” Medium, 23 Oct. 2023, medium.com/@ja_aitsaid/automatic-curriculum-learning-solving-rubiks-cube-beyond-ppo-limitations-3e6489b2ff6f. 

## Code Citation

This repository builds on **DeepCubeA**, originally developed by [Forest Agostinelli et al.](https://github.com/forestagostinelli/DeepCubeA). If you use this work in research or projects, please cite the original paper:

> Agostinelli, F., McAleer, S., Shmakov, A., & Baldi, P. (2019). *Solving the Rubik’s cube with deep reinforcement learning and search*. Nature Machine Intelligence, 1(8), 356–363. [DOI:10.1038/s42256-019-0070-z](https://doi.org/10.1038/s42256-019-0070-z)

The RubiksCubeGym environment is also used for data generation that was developed by [Hukmani et al](https://github.com/DoubleGremlin181/RubiksCubeGym):
> Hukmani, K., Kolekar, S., & Vobugari, S. (2021). *Solving Twisty Puzzles Using Parallel Q-learning.* Engineering Letters, 29(4).

Other github repos that have been referenced during implementation are:
https://github.com/WillPalaia/RubiksCubeSolver
https://github.com/DoubleGremlin181/RubiksCubeRL
*Noted in individual files which have been modified after drawing inspiration from one of the github
