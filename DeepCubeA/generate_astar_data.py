import gymnasium as gym  
from DeepCubeA.environments import cube_222
from tqdm import tqdm
from collections import deque
import time
import pickle

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

# (1) Generate Pickle File
# 2x2x2 Rubik's Cube
def generate_222_data(env, num_states=1000, scramble_cap=14):
    data = {"states": [],
            "times": [],
            "solutions": [],
            "num_nodes_generated": []}
    states, scramble_nums = env.generate_states(num_states, (2, scramble_cap))
    for i, state in tqdm(enumerate(states)):
        solution, num_nodes_generated, solve_time = bfs(env, state)
        data["states"].append(state)
        data["times"].append(solve_time)
        data["solutions"].append(solution)
        data["num_nodes_generated"].append(num_nodes_generated)
    
    return data

def save_dict_data(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data['states'])} states into {fname}")

if __name__ == "__main__":
    env = cube_222.Rubiks2()
    data = generate_222_data(env)
    save_dict_data(data, fname = "data/cube222/train/data_0.pkl")
