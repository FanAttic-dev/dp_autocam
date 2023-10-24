from abc import ABC, abstractmethod

from utils.protocols import HasStats


class Algo(ABC, HasStats):
    @abstractmethod
    def update_by_bbs(self, bbs):
        ...
