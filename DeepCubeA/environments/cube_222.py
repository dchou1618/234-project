import numpy as np
from typing import List, Tuple
import torch.nn as nn

from environment_abstract import Environment, State
from utils.pytorch_models import ResnetModel


class Rubiks2State(State):
    """
    Rubiks2State holds 24-state representation of the cube.
    """
    __slots__ = ['cube', '_hash']

    def __init__(self, cube: np.ndarray):
        self.cube = cube
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.cube.tostring())
        return self._hash

    def __eq__(self, other):
        # Two states are equal if their 24-element arrays match
        return np.array_equal(self.cube, other.cube)

    def _set_state(self, new_state):
        assert isinstance(new_state, np.ndarray), "Not an ndarray"
        assert new_state.shape==(24,), f"Not right shape: {new_state.shape}"
        self.cube = new_cube.copy()
        self._hash = None


def _move_2x2(state_array: np.ndarray, face: str, reps: int = 1) -> np.ndarray:
    """
    From rubiks_cube_222 method DoubleGremlin181/RubiksCubeGym -> RubiksCube222Env.move 
    Applies a single 2x2 face turn to the 24-element numpy array that
    represents the cube, repeated 'reps' times.
    F, R, U
    """
    new_array = state_array.copy()


    def move_once(cube: np.ndarray, move_side: str):
        # Effectively the same moves as scramble - iterating over FRU.
        if move_side == "F":
            side_old = np.array([2, 3, 13, 5, 21, 20, 8, 16])
            face_old = np.array([[6, 7], [14, 15]])
        elif move_side == "R":
            side_old = np.array([1, 3, 7, 15, 21, 23, 18, 10])
            face_old = np.array([[8, 9], [16, 17]])
        elif move_side == "U":
            side_old = np.array([6, 7, 8, 9, 10, 11, 4, 5])
            face_old = np.array([[0, 1], [2, 3]])
        else:
            raise ValueError(f"Unsupported face '{move_side}' in _move_2x2")

        side_new = np.roll(side_old, -2) 
        face_old_flat = face_old.flatten()
        face_new_flat = np.rot90(face_old, 3).flatten()

        cube[side_old], cube[face_old_flat] = cube[side_new], cube[face_new_flat]

    # Apply 'reps' times
    for _ in range(reps):
        move_once(new_array, face)

    return new_array


class Rubiks2(Environment):
    """
    DeepCubeA-style environment for the 2x2 Rubik's cube.
    Define 6 moves total: F, F', R, R', U, U'.
    """
    # Our forward moves:
    #   0 => F
    #   1 => F' (3 reps)
    #   2 => R
    #   3 => R' (3 reps)
    #   4 => U
    #   5 => U' (3 reps)
    # You could define 3 moves if you prefer, but then you must handle
    # inversion carefully in `prev_state`.

    move_map = {
        0: ("F", 1),
        1: ("F", 3),
        2: ("R", 1),
        3: ("R", 3),
        4: ("U", 1),
        5: ("U", 3),
    }

    # For generating prev_state, we just invert each move:
    # If next_state was "F", then prev_state is "F'". If next_state was "F'",
    # then prev_state is "F", etc.
    # We'll just compute this as well:
    move_inverses = {
        0: 1,  # F -> F'
        1: 0,  # F' -> F
        2: 3,  # R -> R'
        3: 2,  # R' -> R
        4: 5,  # U -> U'
        5: 4,  # U' -> U
    }

    def __init__(self):
        super().__init__()
        # By default, the environment_abstract sets self.dtype = float, so set to uint8
        self.dtype = np.uint8
        self.fixed_actions = True
        self.cube_len = 2
        self._solved_state_array = np.arange((self.cube_len ** 2) * 6, dtype=self.dtype)


    def next_state(self,
                   states: List[Rubiks2State],
                   action: int) -> Tuple[List[Rubiks2State], List[float]]:
        """
        Applies the # action move to each state in 'states'. 
        Returns the new states plus a list of costs (always 1.0).
        """
        face, reps = self.move_map[action]

        new_states = []
        costs = []
        for s in states:
            next_cube = _move_2x2(s.cube, face, reps)
            new_states.append(Rubiks2State(next_cube))
            costs.append(1.0)

        return new_states, costs

    def prev_state(self, states: List[Rubiks2State], action: int) -> List[Rubiks2State]:
        """
        The 'inverse' of applying 'action'. If action was F, we do F' (and vice versa).
        """
        inverse_action = self.move_inverses[action]
        # We only need the states, not the cost, so slice out [0]
        return self.next_state(states, inverse_action)[0]

    def generate_goal_states(self, num_states: int) -> List[Rubiks2State]:
        """
        Return a list of 'num_states' solved states. 
        In a 2x2, the solved state is always the same, so we can just repeat.
        """
        return [Rubiks2State(self._solved_state_array.copy())
                for _ in range(num_states)]

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Rubiks2State], List[int]]:
        states = self.generate_goal_states(num_states)
        # 0 to back_max
        scramble_nums = np.random.randint(backwards_range[0], backwards_range[1] + 1, size=num_states)
        
        for i, num_scrambles in enumerate(scramble_nums):
            for _ in range(num_scrambles):
                action = np.random.choice(list(self.move_map.keys()))
                states[i] = self.next_state([states[i]], action)[0][0]
        
        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        expanded_states = []
        transition_costs = []
        
        for state in states:
            successors = []
            costs = []
            for action in self.move_map.keys():
                next_states, cost = self.next_state([state], action)
                successors.append(next_states[0])
                costs.append(cost[0])
            expanded_states.append(successors)
            transition_costs.append(np.array(costs))
        
        return expanded_states, transition_costs

    def is_solved(self, states: List[Rubiks2State]) -> np.ndarray:
        """
        For each state, check if it equals the solved arrangement [0..23].
        Return a boolean np.array.
        """
        solved_flags = []
        for s in states:
            solved_flags.append(np.array_equal(s.cube, self._solved_state_array))
        return np.array(solved_flags, dtype=bool)

    def state_to_nnet_input(self, states: List[Rubiks2State]) -> List[np.ndarray]:
        """
        Normalizes the arrangement of the 0-23 array state representation 
        that gets passed as input into get_nnet_model.
        """
        batch = np.stack([s.cube for s in states], axis=0).astype(np.float32)
        batch /= 23.0
        return [batch]

    def get_num_moves(self) -> int:
        """
        We have 6 discrete moves: F, F', R, R', U, U'.
        """
        return len(self.move_map)

    def get_nnet_model(self) -> nn.Module:
        """
        Returns a neural network. For a 2x2, input size is 24 - 
        number of cells facing outward on the cube.
        """
        in_dim: int = (self.cube_len ** 2) * 6

        model = nn.Sequential(
            nn.Linear(in_dim, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1)
        )
        return model