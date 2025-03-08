from environment_abstract import Environment, State
import numpy as np
from torch import nn

class PyraminxState(State):
    __slots__ = ['pyramid', '_hash']

    def __init__(self, pyramid):
        self.pyramid = pyramid
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.pyramid.tostring())
        return self._hash

    def __eq__(self, other):
        return np.array_equal(self.pyramid, other.pyramid)

    def _set_state(self, new_state):
        self.pyramid = new_state.copy()
        self._hash = None

    def __repr__(self):
        return f"Pyraminx({self.pyramid})"


class Pyraminx(Environment):
    moves = ['L', 'R', 'U', 'B']
    moves_rev = ["L'", "R'", "U'", "B'"]

    def __init__(self):
        super().__init__()
        self.ACTION_MAP = {idx: (self.__class__.moves[idx], None) for idx in range(len(self.__class__.moves))}
        self.MOVE_MAP = {idx: (self.__class__.moves[idx], 1) for idx in range(len(self.__class__.moves))}
        self.goal_state: np.ndarray = np.arange(0, 36, dtype=self.dtype)

    def _move_pyraminx(self, state_arr, action, move_type, reps):
        curr_arr = state_arr.copy()

        def move(state_arr, action, move_type):
            repetitions = dict({None: 1, "'": 2})[move_type]

            if action.isupper():
                if action == "L":
                    layer_cubies_old = np.array([14, 22, 23, 11, 12, 13, 29, 28, 32])
                elif action == "R":
                    layer_cubies_old = np.array([16, 24, 23, 29, 30, 34, 19, 18, 17])
                elif action == "U":
                    layer_cubies_old = np.array([2, 3, 13, 14, 15, 16, 17, 7, 8])
                elif action == "B":
                    layer_cubies_old = np.array([11, 1, 2, 8, 9, 19, 34, 33, 32])

                layer_cubies_new = np.roll(layer_cubies_old, -3 * repetitions)
                np.put(state_arr, layer_cubies_old, state_arr[layer_cubies_new])

                action = action.lower()

            if action == "l":
                vertex_cubies_old = np.array([20, 27, 21])
            elif action == "r":
                vertex_cubies_old = np.array([25, 31, 26])
            elif action == "u":
                vertex_cubies_old = np.array([4, 5, 6])
            elif action == "b":
                vertex_cubies_old = np.array([0, 10, 35])

            vertex_cubies_new = np.roll(vertex_cubies_old, -1 * repetitions)
            np.put(state_arr, vertex_cubies_old, state_arr[vertex_cubies_new])
        
        for _ in range(reps):
            move(curr_arr, action, move_type)

        return curr_arr

    def next_state(self, states, action: int):

        _, repetitions = self.MOVE_MAP[action]
        action, move_type = self.ACTION_MAP[action]
        costs = []
        new_states = []
        for state in states:
            new_states.append(PyraminxState(self._move_pyraminx(state.pyramid, action, move_type, repetitions)))
            costs.append(1.0)
        return new_states, costs


    def prev_state(self, states, action):
        """
        Reverses a move to get the previous state.
        """
        move = self.moves[action]
        move_rev_idx = self.moves.index(self.moves_rev[action])

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False):

        if np_format:
            goal_np = np.expand_dims(self.goal_state.copy(), 0)
            solved_states = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states = [PyraminxState(self.goal_state.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states):

        states_np = np.stack([state.pyramid for state in states], axis=0)
        return np.all(states_np == self.goal_state, axis=1)

    def state_to_nnet_input(self, states):
        """
        Converts states to a format suitable for a neural network.
        """
        states_np = np.stack([state.pyramid for state in states], axis=0)
        representation_np = states_np / (len(self.goal_state)-1)
        representation_np = representation_np.astype(np.float32)

        return [representation_np]

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self):
        """
        Returns a neural network model for cost-to-go estimation.
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

    def generate_states(self, num_states, backwards_range):
        """
        Generates scrambled states by applying random moves.
        """

        states_np = self.generate_goal_states(num_states, np_format=True)

        scramble_nums: np.array = np.random.randint(backwards_range[0], backwards_range[1]+1, size=num_states)
        for i in range(num_states):
            for _ in range(scramble_nums[i]):
                move = np.random.choice(self.moves)
                move_int = self.moves.index(move)
                _, repetitions = self.MOVE_MAP[move_int]
                action, move_type = self.ACTION_MAP[move_int]
                states_np[i] = self._move_pyraminx(states_np[i], action, move_type, repetitions)

        states = [PyraminxState(x) for x in list(states_np)]
        return states, scramble_nums.tolist()
