from rubiks_cube_222_multibinary import RubiksCube222EnvB
import numpy as np

# modified from: https://github.com/DoubleGremlin181/RubiksCubeGym
# wrapper to convert state space from discrete to multibinary

class RubiksCube222EnvLBLPPOB(RubiksCube222EnvB):
    def __init__(self):
        super(RubiksCube222EnvLBLPPOB, self).__init__()
        self.FL = None
        self.OLL = None
        self.time_limit = None
        self.scrambles = 1
        self.max_moves = 50
        self.current_moves = 0
        solved = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                             dtype=np.uint8)
        solved_obs = [COLOR_IDX[TILE_MAP[i]] for i in solved]
        self.solved_obs = np.eye(6)[solved_obs].flatten()

    def check_solved(self):
        if self.cube_reduced == "WWWWOOGGRRBBOOGGRRBBYYYY":
            return True

    def manhattan_distance(self):
        obs = np.eye(6)[self.convert(self.cube)].flatten()
        return  np.sum(np.abs(self.solved_obs - obs))

    def reward(self):
        done = False

        if self.check_solved():
            reward = 60
            done = True
            return reward, done

        reward = -1 - 0.1*self.manhattan_distance()
        
        self.current_moves += 1
        if self.current_moves >= self.max_moves:
            reward = -10
            done = True
        return reward, done


    def step(self, action):
        obs, reward, done, _, info = super().step(action=action)
        observation = np.eye(6)[self.convert(obs)].flatten()

        return observation, reward, done, False, info

    def convert(self, obs): # convert discrete state space to multibinary
        new_obs = [COLOR_IDX[TILE_MAP[i]] for i in obs]
        return new_obs
    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset_scramble(seed=seed, num_scramble=self.scrambles)
        self.current_moves = 0
        observation = np.eye(6)[self.convert(obs)].flatten()
        return observation, info
    
    def reset_scramble(self, num_scramble, *, seed=None, options=None):
        obs, info = super().reset_scramble(seed=seed, num_scramble=num_scramble)
        observation = np.eye(6)[self.convert(obs)].flatten()
        return observation, info
    


TILE_MAP = {
    0: 'W', 1: 'W', 2: 'W', 3: 'W',
    4: 'O', 5: 'O', 6: 'G', 7: 'G', 8: 'R', 9: 'R', 10: 'B', 11: 'B',
    12: 'O', 13: 'O', 14: 'G', 15: 'G', 16: 'R', 17: 'R', 18: 'B', 19: 'B',
    20: 'Y', 21: 'Y', 22: 'Y', 23: 'Y'

}

COLOR_MAP = {
    'W': (255, 255, 255),
    'O': (255, 165, 0),
    'G': (0, 128, 0),
    'R': (255, 0, 0),
    'B': (0, 0, 255),
    'Y': (255, 255, 0)
}

COLOR_IDX = {
    'W': 0,
    'R': 1,
    'Y': 2,
    'O': 3, 
    'B': 4,
    'G': 5,
}

ACTION_MAP = {0: ("F", None), 1: ("R", None), 2: ("U", None)}
