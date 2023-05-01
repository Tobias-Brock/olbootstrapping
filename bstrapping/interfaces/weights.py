from abc import abstractmethod

import numpy as np


class Weights:
    """Generic interface for weights used in the weighted bootstrap algorithm

    """

    @abstractmethod
    def __call__(self, ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            1 dimensional array of size of numbers of samples

        """
        raise NotImplementedError
