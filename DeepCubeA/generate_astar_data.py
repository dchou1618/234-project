import gymnasium as gym  
from DeepCubeA.environments import cube_222, skewb, pyraminx
from tqdm import tqdm
from collections import deque
import time
import pickle
from utils import nnet_utils
from search_methods.gbfs import GBFS
import numpy as np

# (3) gbfs - run greedy best first search using heuristic trained using avi.py

def gbfs_helper(env, state, heuristic_fn, max_solve_steps = 100):
    start = time.time()
    gbfs = GBFS([state], env)
    for _ in range(max_solve_steps):
        gbfs.step(heuristic_fn)
        if gbfs.get_is_solved()[0]:
            break
    traj = gbfs.get_trajs()[0]
    solution = [step[0] for step in traj]
    solve_time = time.time() - start
    num_nodes_generated = gbfs.get_num_steps()[0]
    return solution, num_nodes_generated, solve_time


# (1) Generate Pickle File
# 2x2x2 Rubik's Cube
def generate_data(env, pt_fname="../DeepCubeA/saved_models/cube222/current/model_state_dict.pt", 
                           solver=gbfs_helper, 
                           num_states=1400, 
                           scramble_cap=14):
    data = {"states": [],
            "times": [],
            "solutions": [],
            "num_nodes_generated": [],
            "scrambles": []}
    nnet = env.get_nnet_model()
    device, _, _ = nnet_utils.get_device()
    nnet = nnet_utils.load_nnet(pt_fname, nnet, device=device)
    nnet.eval()
    nnet.to(device)

    heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env, clip_zero=False)
    for scramble_num in tqdm(range(1,scramble_cap+1)):
        for samples in range(100): 
            states, _ = env.generate_states(1, (scramble_num, scramble_num))
            state = states[0]
            while state.is_solved():
                states, _ = env.generate_states(1, (scramble_num, scramble_num))
                state = states[0]

            solution, num_nodes_generated, solve_time = solver(env, state, heuristic_fn=heuristic_fn)
            data["states"].append(state)
            data["times"].append(solve_time)
            data["solutions"].append(solution)
            data["num_nodes_generated"].append(num_nodes_generated)
            data["scrambles"].append(scramble_num)
        
    return data


# (2) Helper - Run Ground Truth Solver on Scrambled State to obtain solution.
def bfs(env, state):
    start = time.time()
    visited = set()
    # tuple of current state, solution (list of actions)
    frontier = deque([(state, [])])
    num_nodes_generated = 0
    while len(frontier) > 0:
        state, sol = frontier.popleft()
        num_nodes_generated += 1
        if env.is_solved([state])[0]:
            return sol, num_nodes_generated, time.time()-start
        for action in range(env.get_num_moves()):
            # a singleton list containing only state
            next_states, _ = env.next_state([state], action)
            next_state = next_states[0]
            next_state_hash = hash(next_state)
            if next_state_hash not in visited:
                visited.add(next_state_hash)
                frontier.append((next_state, sol+[env.move_map[action]]))

    raise RuntimeError("BFS did not reach a solution from scrambled state.")


def save_dict_data(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data['states'])} states into {fname}")

if __name__ == "__main__":
    for j in range(1,6):
        env = cube_222.Rubiks2()
        data = generate_data(env, scramble_cap=14)
        save_dict_data(data, fname = f"../DeepCubeA/data/cube222/train/data_{j}.pkl")

        env = skewb.SkewB()
        data = generate_data(env, pt_fname="../DeepCubeA/saved_models/skewb/current/model_state_dict.pt", 
                            scramble_cap=10)
        save_dict_data(data, fname = f"../DeepCubeA/data/skewb/train/data_{j}.pkl")

        env = pyraminx.Pyraminx()
        data = generate_data(env, pt_fname="../DeepCubeA/saved_models/pyraminx/current/model_state_dict.pt",
                            scramble_cap=11)
        save_dict_data(data,
                    fname = f"../DeepCubeA/data/pyraminx/train/data_{j}.pkl")
