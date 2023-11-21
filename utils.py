from dataclasses import dataclass
from typing import List


@dataclass
class Gaussian():
    def __init__(self, theta, a, b) -> None:
        self.__theta = theta
        self.__a = a
        self.__b = b
    
    @property
    def theta(self):
        return self.__theta
    
    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b

@dataclass
class GaussianSignal():
    def __init__(self, gaussians: List[Gaussian]) -> None:
        self.__gaussians = gaussians

    @property
    def gaussians(self):
        return self.__gaussians
