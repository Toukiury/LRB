from abc import ABC, abstractmethod
import random

class Evictor(ABC):
    @abstractmethod
    def evict(self, candidates):
        pass

class DummyEvictor(ABC):
    def evict(self, candidates):
        return 0

class MaxEvictor(Evictor):
    def evict(self, candidates):
        if not candidates:  # 如果候选列表为空
            # 如果没有候选项，返回0作为默认值
            return 0
        return max(candidates, key=lambda x: x[1])[0]

ReuseDistanceEvictor = MaxEvictor

class BinaryEvictor(Evictor):
    def evict(self, candidates):
        if not candidates:  # 如果候选列表为空
            # 如果没有候选项，返回0作为默认值
            return 0
        indices_with_1 = [i for i, pred in candidates if pred == 1]
        if indices_with_1:
            chosen_index = random.choice(indices_with_1)
        else:
            chosen_index = random.choice(candidates)[0]
        return chosen_index

class MinEvictor(Evictor):
    def evict(self, candidates):
        if not candidates:  # 如果候选列表为空
            # 如果没有候选项，返回0作为默认值
            return 0
        return min(candidates, key=lambda x: x[1])[0]

LRUEvictor = MinEvictor

class MarkerEvictor(Evictor):
    def evict(self, candidates):
        if not candidates:  # 如果候选列表为空
            # 如果没有候选项，返回0作为默认值
            return 0
        # 1 marked, 0 unmarked
        indices_with_0 = [i for i, mark in candidates if mark == 0]
        if indices_with_0:
            chosen_index = random.choice(indices_with_0)
        else:
            # 如果所有页面都被标记，随机选择一个
            chosen_index = random.choice([i for i, _ in candidates])

        return chosen_index

class RandEvictor(Evictor):
    def evict(self, candidates):
        if not candidates:  # 如果候选列表为空
            # 如果没有候选项，返回0作为默认值
            return 0
        return random.choice([i for i, _ in candidates])
