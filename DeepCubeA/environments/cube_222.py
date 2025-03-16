import numpy as np
from typing import List, Tuple
import torch.nn as nn

from environment_abstract import Environment, State
from utils.pytorch_models import ResnetModel


class Rubiks2State(State):
    """
    Rubiks2State holds a 24-state representation of the cube.
    Stores cube and hash value for Rubiks2 environment.
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
        self.cube = new_state.copy()
        self._hash = None


def _move_2x2(state_array: np.ndarray, face: str, reps: int = 1) -> np.ndarray:
    """
    _move_2x2: adapted from the rubiks_cube_222 method DoubleGremlin181/RubiksCubeGym 
    -> RubiksCube222Env.move. Applies one of three actions F,R,U to a 2x2 face turn 
    on 24-element numpy array.
    """
    new_array = state_array.copy()


    def move_once(cube: np.ndarray, move_side: str):
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

    for _ in range(reps):
        move_once(new_array, face)

    return new_array


class Rubiks2(Environment):
    """
    2x2 Rubik's cube environment from RubiksCubeGym adapted for DeepCubeA usage.
    We define 6 moves total: F, F', R, R', U, U'.
    Reverse moves are repeated 3 times in the same way that DoubleGremlin181/RubiksCubeGym
    defines them.
    """

    move_map = {
        0: ("F", 1),
        1: ("F", 3),
        2: ("R", 1),
        3: ("R", 3),
        4: ("U", 1),
        5: ("U", 3),
    }
    move_inverses = {
        0: 1,  
        1: 0,  
        2: 3, 
        3: 2, 
        4: 5,  
        5: 4, 
    }

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.fixed_actions = True
        self.cube_len = 2
        self._solved_state_array = np.arange((self.cube_len ** 2) * 6, dtype=self.dtype)


    def next_state(self,
                   states: List[Rubiks2State],
                   action: int) -> Tuple[List[Rubiks2State], List[float]]:
        """
        next state: applies the specified action to each state in states. 
        :param states: list of rubiks2 state objects. Each object contains attribute cube.
        :param action: an integer denoting one of the six actions.
        :return: new_states, costs - the next states and a list of costs.
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
        prev_state: retrieve the previous state by applying the reverse action.
        :param states: list of rubiks2 state objects.
        :param action: integer denoting one of six actions.
        :return: prev_states, costs - retrieved by calling next_state on the reverse action indexed
        in move_inverses.
        """
        inverse_action = self.move_inverses[action]
        return self.next_state(states, inverse_action)[0]

    def generate_goal_states(self, num_states: int) -> List[Rubiks2State]:
        """
        generate_goal_states: generates a list of solved states to apply reverse moves to.
        :param num_states: number of goal states to generate
        :return: the list of rubiks2states of length num_states
        """
        return [Rubiks2State(self._solved_state_array.copy())
                for _ in range(num_states)]

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Rubiks2State], List[int]]:
        """
        generate_states: takes in num_states, backwards_range and returns a new list of states where 
        each solved state is scrambled an arbitrary amount of times uniformly within the backwards range.
        :param num_states: number of states to generate
        :param backwards_range: range of backward moves from solved states that are allowable
        :return: states, scramble_nums - the list of scrambled states and number of scrambles per list.
        """
        states = self.generate_goal_states(num_states)
        scramble_nums = np.random.randint(backwards_range[0], backwards_range[1] + 1, size=num_states)
        
        for i, num_scrambles in enumerate(scramble_nums):
            for _ in range(num_scrambles):
                action = np.random.choice(list(self.move_map.keys()))
                states[i] = self.next_state([states[i]], action)[0][0]
        
        return states, scramble_nums.tolist()

    def expand(self, states: List[Rubiks2State]) -> Tuple[List[List[Rubiks2State]], List[np.ndarray]]:
        """
        expand: taking in a list of states, return a list of lists displaying all possible next states 
        after iterating over all possible actions and returning a list of list of states along with the 
        list of arrays of costs.
        :param states: list of states
        :return: expanded_states, transition_costs - for each state entry, this is replaced with the list of 
        possible next states.
        """
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
        is_solved: taking in a list of rubiks2states, return whether or not cube is solved.
        :param states: list of states
        :return: solved_flags - whether each state is solved or not, matching the _solved_state_array.
        """
        solved_flags = []
        for s in states:
            solved_flags.append(np.array_equal(s.cube, self._solved_state_array))
        return np.array(solved_flags, dtype=bool)

    def state_to_nnet_input(self, states: List[Rubiks2State]) -> List[np.ndarray]:
        """
        state_to_nnet_input: scales the states array by the solved state array length-1
        :param states: 
        :return:
        """
        batch = np.stack([s.cube for s in states], axis=0).astype(np.float32)
        batch /= (len(self._solved_state_array)-1)
        return [batch]

    def get_num_moves(self) -> int:
        return len(self.move_map)

    def get_nnet_model(self) -> nn.Module:
        """
        get_nnet_model: returns the cost-to-go neural network.
        """
        in_dim: int = (self.cube_len ** 2) * 6

        model = nn.Sequential(
            nn.Linear(in_dim, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1)
        )
        return model