import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j),(i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N))
        # TODO:
        return T

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        M, N = self.grid.shape
        O = np.zeros((16, M * N))
        # TODO:
        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO:
        pass

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """
        # TODO:
        pass

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO:
        pass

    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO:
        pass

    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        # TODO:
        pass
