from abc import abstractmethod


class Algo:
    @abstractmethod
    def update_by_bbs(self, bbs):
        ...