from rubiks_cube_222 import RubiksCube222Env

# modified from: https://github.com/DoubleGremlin181/RubiksCubeGym
# removed FL_POS and OLL_POS check
class RubiksCube222EnvLBL(RubiksCube222Env):
    def __init__(self):
        super(RubiksCube222EnvLBL, self).__init__()
    
    def check_solved(self):
        if self.cube_reduced == "WWWWOOGGRRBBOOGGRRBBYYYY":
            return True

    def reward(self):
        done = False

        if self.check_solved():
            reward = 60
            done = True
            return reward, done

        reward = -1

        return reward, done

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
    
    def reset_scramble(self, num_scramble, *, seed=None, options=None):
        obs, info = super().reset_scramble(seed=seed, num_scramble=num_scramble)
        return obs, info
    
