from abc import abstractmethod

from utils.protocols import HasStats


class Algo(HasStats):
    @abstractmethod
    def update_by_bbs(self, bbs):
        ...
