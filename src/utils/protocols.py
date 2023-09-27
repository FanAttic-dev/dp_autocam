from typing import Protocol


class HasStats(Protocol):
    def get_stats(self) -> dict:
        ...
