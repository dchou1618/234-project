from environment_abstract import Environment, State
import numpy as np
from torch import nn
from typing import List, Tuple

class SkewBState(State):
    __slots__ = ['cube', '_hash']

    def __init__(self, cube: np.ndarray):
        self.cube = cube
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.cube.tostring())
        return self._hash

    def __eq__(self, other):
        return np.array_equal(self.cube, other.cube)

    def _set_state(self, new_state):
        self.cube = new_state.copy()
        self._hash = None

    def __repr__(self):
        return f"SkewB({self.cube})"


class SkewB(Environment):
    """
    Defined moves LRUB consistent with that of https://github.com/DoubleGremlin181/RubiksCubeGym
    """
    moves =  ['L', 'R', 'U', 'B']
    moves_rev = ['L', 'R', 'U', 'B']

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 1
        self.ACTION_MAP = {idx: (self.__class__.moves[idx], None) for idx in range(len(self.__class__.moves))}
        self.MOVE_MAP = {idx: (self.__class__.moves[idx], 1) for idx in range(len(self.__class__.moves))}
        self.goal_state: np.ndarray = np.arange(0, 30, dtype=self.dtype) 

    def _move_np(self, states_np: np.ndarray, action: int, move_type: int, reps: int) -> np.ndarray:
        """
        _move_np: takes a state array of 30 elements and 
        repeats each action 'reps' number of times. MOVE_MAP 
        only defines moves that are run once.
        :param state_arr: integer numpy array of elements in the range
        0 to 29, inclusive.
        :param action: an integer from 0 to 3 denoting one of the LRUB moves.
        :param move_type: move_type determines the number of 'repetitions' to apply
        the action.
        :param reps: the number of times to apply each move that is repeated 'repetitions'
        many times.
        :return: updated state_arr after moves are applied.
        """
        curr_arr = states_np.copy()

        def move(states, action, move_type):
            repetitions = dict({None: 1, "'": 2})[move_type]

            if action == "L":
                layer_cubies_old = np.array([10, 12, 14, 8, 7, 6, 26, 27, 28])
                vertex_cubies_old = np.array([13, 9, 25])
                side_cubies_old = np.array([3, 24, 18])

            elif action == "R":
                layer_cubies_old = np.array([16, 17, 18, 26, 27, 28, 24, 22, 20])
                vertex_cubies_old = np.array([19, 29, 23])
                side_cubies_old = np.array([1, 14, 8])

            elif action == "U":
                layer_cubies_old = np.array([3, 2, 1, 20, 22, 24, 8, 7, 6])
                vertex_cubies_old = np.array([0, 21, 5])
                side_cubies_old = np.array([16, 28, 10])

            elif action == "B":
                layer_cubies_old = np.array([21, 22, 23, 29, 27, 25, 9, 7, 5])
                vertex_cubies_old = np.array([24, 28, 8])
                side_cubies_old = np.array([0, 19, 13])
            else:
                raise ValueError(f"Unrecognized Action: {action}")

            layer_cubies_new = np.roll(layer_cubies_old, -3 * repetitions)
            vertex_cubies_new = np.roll(vertex_cubies_old, -1 * repetitions)
            side_cubies_new = np.roll(side_cubies_old, -1 * repetitions)

            np.put(states, layer_cubies_old, states[layer_cubies_new])
            np.put(states, vertex_cubies_old, states[vertex_cubies_new])
            np.put(states, side_cubies_old, states[side_cubies_new])

        for _ in range(reps):
            move(curr_arr, action, move_type)

        return curr_arr

    def next_state(self, states: List[SkewBState], action: int) -> Tuple[List[SkewBState], List[float]]:
        """
        next_state: takes a state and action and obtains the next state from each state in the 'states'
        variable.
        :param states: a list of skewb states, each containing skewb array attribute.
        :param action: an integer 0 to 3 denoting the possible moves to apply
        :return: next states, costs
        """
        _, repetitions = self.MOVE_MAP[action]
        action, move_type = self.ACTION_MAP[action]
        costs = []
        new_states = []
        for state in states:
            new_states.append(SkewBState(self._move_np(state.cube, action, move_type, repetitions)))
            costs.append(1.0)
        return new_states, costs

    def prev_state(self, states: List[SkewBState], action: int) -> Tuple[List[SkewBState], List[float]]:
        """
        prev_state: applies inverse move to obtain the previous state.
        :param states: list of skewb states.
        :param action: an integer from 0 to 3
        :return: previous states, costs
        """
        return self.next_state(states, action)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> List[SkewBState]:
        """
        generate_goal_states: generates a list of solved states to apply reverse moves to.
        :param num_states: number of goal states to generate
        :return: solved_states - the list of skewbstates of length num_states
        """
        if np_format:
            goal_np = np.expand_dims(self.goal_state.copy(), 0)
            solved_states = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states = [SkewBState(self.goal_state.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[SkewBState]) -> np.ndarray:
        """
        is_solved: taking in a list of skewb states, return whether or not cube is solved.
        :param states: list of states
        :return: solved_flags - whether each state is solved or not, matching the goal state.
        """
        states_np = np.stack([state.cube for state in states], axis=0)
        return np.all(states_np == self.goal_state, axis=1)

    def state_to_nnet_input(self, states: List[SkewBState]) -> List[np.ndarray]:
        """
        state_to_nnet_input: scales the state array entries by the length of the goal state-1
        to lead to more stabilized NN training.
        """
        states_np = np.stack([state.cube for state in states], axis=0)
        representation_np = states_np / (len(self.goal_state)-1)
        representation_np = representation_np.astype(np.float32)

        return [representation_np]

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        """
        get_nnet_model: returns the cost-to-go neural network.
        """
        in_dim = len(self.goal_state)
        layers = [nn.Linear(in_dim, 5000), 
                  nn.LeakyReLU()]
        for i in range(2):
            layers.append(nn.Linear(5000, 5000))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(5000, 1))
        model = nn.Sequential(*layers)
        return model

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[SkewBState], List[int]]:
        """
        generate_states: takes in num_states, backwards_range and returns a new list of states where 
        each solved state is scrambled an arbitrary amount of times uniformly within the backwards range.
        :param num_states: number of states to generate
        :param backwards_range: range of backward moves from solved states that are allowable
        :return: states, scramble_nums - the list of scrambled states and number of scrambles per list.
        """
        states_np = self.generate_goal_states(num_states, np_format=True)

        scramble_nums: np.array = np.random.randint(backwards_range[0], backwards_range[1]+1, size=num_states)
        for i in range(num_states):
            for _ in range(scramble_nums[i]):
                move = np.random.choice(self.moves)
                move_int = self.moves.index(move)
                _, repetitions = self.MOVE_MAP[move_int]
                action, move_type = self.ACTION_MAP[move_int]
                states_np[i] = self._move_np(states_np[i], action, move_type, repetitions)

        states: List[SkewBState] = [SkewBState(x) for x in list(states_np)]
        return states, scramble_nums.tolist()