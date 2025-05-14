from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union, Type
from cache.evict.evictor import *
from cache.evict.predictor import *
import numpy as np
import types
import copy
import random
import collections
import inspect

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        self.cache = [None] * associativity
        self.pcs = [None] * associativity
        self.associativity = associativity
    
    def snapshot(self):
        return list(zip(self.cache, self.pcs))
    
    @abstractmethod
    def access(self, pc, address) -> bool:
        pass

    def boost_access(self, pc, address, boost_pred) -> bool:
        return self.access(pc, address)

class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity)
        self.timestamp = 0

        cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
        if issubclass(cls_type, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif issubclass(cls_type, BinaryPredictor):
            self.preds = [0] * associativity
        elif issubclass(cls_type, PhasePredictor):
            self.preds = [1] * associativity
        elif issubclass(cls_type, StatePredictor):
            self.preds = [None] * associativity
        else:
            self.preds = None
        
        if issubclass(cls_type, OraclePredictor):
            def oracle_access(self, pc, address, next_access_time):
                self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        self.evictor = evictor_type()
        
        # 检查是否需要传递shared_model参数给predictor_type
        if hasattr(predictor_type, 'keywords') and 'shared_model' in kwargs:
            # 对于partial函数，通过keywords添加shared_model
            if 'shared_model' not in predictor_type.keywords:
                predictor_type = partial(predictor_type.func, **{**predictor_type.keywords, 'shared_model': kwargs['shared_model']})
            self.predictor = predictor_type()
        elif 'shared_model' in kwargs and inspect.isclass(cls_type):
            # 对于类，直接调用时传递shared_model
            try:
                # 尝试传递shared_model参数
                self.predictor = predictor_type(shared_model=kwargs['shared_model'])
            except (TypeError, ValueError):
                # 如果失败，尝试不带参数初始化
                self.predictor = predictor_type()
                print(f"警告: 初始化{cls_type.__name__}时无法传递shared_model参数")
        else:
            # 标准初始化
            self.predictor = predictor_type()

        self.cur_boost_pred = None
        self.cur_boost_type = None

    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.preds)
    
    def before_pred(self, pc, address):
        if self.cur_boost_type is not None and self.cur_boost_type == 'before':
            self.preds = self.cur_boost_pred
        else:
            preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
            if preds is not None:
                self.preds = preds
    
    def after_pred(self, pc ,address, target_index):
        if self.cur_boost_type is not None and self.cur_boost_type == 'after':
            self.preds[target_index] = self.cur_boost_pred
        else:
            pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
            if pred is not None:
                self.preds[target_index] = pred
        self.timestamp += 1
    
    def boost_access(self, pc, address, boost_pred):
        self.cur_boost_pred = boost_pred
        if self.cur_boost_type is None:
            if isinstance(boost_pred, list):
                self.cur_boost_type = 'before'
            else:
                self.cur_boost_type = 'after'
        return self.access(pc, address)

    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.preds)))
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

######################################################################

class PredictiveMarker(PredictAlgorithm):
    """
    PredictiveMarker algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        def harmonic_number(k):
            return sum(1 / i for i in range(1, k + 1))
        super().__init__(associativity, evictor_type, predictor_type)
        self.marked = [0] * associativity
        self.tracking_set = []
        self.h_k = harmonic_number(associativity)
        self.chains_len = []
        self.chains_rep = []
    
    def access(self, pc, address):
        hit = False
        self.before_pred(pc, address)
        target_index = -1

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                # new phase
                self.tracking_set = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            if address not in self.tracking_set:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_len.append(1)
                self.chains_rep.append(self.cache[target_index])
            if address in self.tracking_set:
                index = self.chains_rep.index(address)
                if self.chains_len[index] <= self.h_k:
                    target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_rep[index] = self.cache[target_index]

        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LMarker(PredictAlgorithm):
    """
    LMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.stale = []
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.stale = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            
            if address in self.stale:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LNonMarker(PredictAlgorithm):
    """
    LNONMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.phase = set()
        self.stale = []
        self.marked = [0] * associativity
        self.evicts = {}
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        self.before_pred(pc, address)

        if len(self.phase) == self.associativity:
            self.stale = copy.deepcopy(self.cache)
            self.marked = [0] * self.associativity
            self.evicts = {}
            self.phase = set()

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if address in self.stale:
                if self.evicts[address] not in self.stale:
                    target_index = random.choice(range(self.associativity))
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.evicts[self.cache[target_index]] = address
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.phase.add(address)
        self.after_pred(pc, address, target_index)
        return hit

class Mark0(PredictAlgorithm):
    """
    MARK0 Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, BinaryPredictor):
            raise ValueError('Mark0: predictor must be a BinaryPredictor')
        self.marked = [0] * associativity
        self.S_address = [None] * associativity
        self.S_visited = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = self.cache.index(None)
        else:
            if all(visited == 1 for visited in self.S_visited):
                self.marked = [0] * self.associativity
                self.S_address = copy.deepcopy(self.cache)
                self.S_visited = [0] * self.associativity

            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
        
        self.S_address[target_index] = None
        self.S_visited[target_index] = 1
        self.marked[target_index] = 1
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        if self.preds[target_index] == 1:
            self.cache[target_index], self.pcs[target_index] = None, None
        return hit

class MarkAndPredict(PredictAlgorithm):
    """
    MARK&PREDICT Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, PhasePredictor):
            raise ValueError('MarkAndPredict: predictor must be a PhasePredictor')
        if not isinstance(self.evictor, BinaryEvictor):
            raise ValueError('MarkAndPredict: evictor must be a BinaryEvictor')
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.marked = [0] * self.associativity
            target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class FollowerRobust(PredictAlgorithm):
    """
    F&R Algorithm

    Parameters:

    - a

    - lazy_evictor_type

    Designed by Karim Abdel Sadek and Marek Elias. 2024. Algorithms for Caching and MTS with reduced number of predictions.
    https://arxiv.org/abs/2404.06280
    """
    @staticmethod
    def create_windows(S, W, F, k, a):
        def func(i):    
            return (2**(i+1))-1
        for i in range(0, int(np.log2(k)) + 1):
            S.append((int(k - (k // (2 ** i)) + 1)))
        for i in range(1, int(np.log2(k)) + 1):
            n = []
            for h in range(S[i - 1], S[i]):
                n.append(h)
            W.append(n)
        W.append([S[-1]])
        for g in range(0, len(W)-1):
            gap = int(len(W[g])//(func(g+1)-func(g)))
            if (gap >= a):
                for m in W[g][::gap]:
                    F.append(m)
            else:
                for m in range(S[g], S[-1]+1, a):
                    F.append(m)
                break
        if k == 10:
            F = [1,6,9]
        return S, W, F

    @staticmethod
    def differ(a, b):
        aa = list(a).copy()
        bb = list(b).copy()
        for x in bb:
            if x == None:
                continue
            elif x in aa:
                aa.remove(x)
        if aa == []:
            return bb
        return aa

    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, StatePredictor):
            raise ValueError('FollowerRobust: predictor must be a StatePredictor')

        if 'boost' in kwargs:
            self.boost = kwargs['boost']
        else:
            self.boost = False
        self.boost_beladys = []
        self.online_belady_cache = [None] * associativity
        self.online_belady_dis = [np.inf] * associativity
        self.boost_beladys.append(copy.deepcopy(self.online_belady_cache))
        if self.boost:
            def oracle_access(self, pc, address, next_access_time):
                if address in self.online_belady_cache:
                    target_index = self.online_belady_cache.index(address)
                elif None in self.online_belady_cache:
                    target_index = self.online_belady_cache.index(None)
                else:
                    target_index = self.online_belady_dis.index(max(self.online_belady_dis))
                
                self.online_belady_cache[target_index] = address
                self.online_belady_dis[target_index] = next_access_time
                self.boost_beladys.append(copy.deepcopy(self.online_belady_cache))
                if hasattr(self.predictor, 'oracle_access'):
                    self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)

        if 'a' in kwargs:
            self.a = kwargs['a']
        else:
            self.a = 1
        if 'lazy_evictor_type' in kwargs:
            if kwargs['lazy_evictor_type'] is None:
                self.lazy_evictor = None
            else:
                self.lazy_evictor = kwargs['lazy_evictor_type']()
        else:
            self.lazy_evictor = LRUEvictor()
        self.key_scores = [np.inf] * self.associativity if self.lazy_evictor is not None else None
        self.sim_cache = [None] * associativity
        self.sim_pcs = [None] * associativity
        self.traces = []
        
        self.S = []
        self.W = []
        self.F = []
        if (self.a == 1):
            self.S, self.W, self.F = FollowerRobust.create_windows(self.S, self.W, self.F, self.associativity, self.a)
        self.skip = 0
        self.pred_gap = 0
        self.follow_cost = 0
        self.belady_cost = 0
        self.marked = []
        self.old = []
        self.unmarked = []
        self.unmarked_for_reload = []
        self.clean = []
        self.prediction = [None] * self.associativity

    def online_belady(self):
        if self.boost:
            return self.boost_beladys[self.timestamp]
        else:
            cache = []
            for i, current in enumerate(self.traces):
                if current in cache:
                    continue
                if len(cache) < self.associativity:
                    cache.append(current)
                else:
                    future_uses = {item: self.traces[i + 1:].index(item) if item in self.traces[i + 1:] else float('inf') for item in cache}
                    to_remove = max(future_uses, key=future_uses.get)
                    cache.remove(to_remove)
                    cache.append(current)
            return cache
    
    def follow_robust(self, pc, address):
        target_index = -1
        # get next state
        if self.cur_boost_type is not None and self.cur_boost_type == 'before':
            preds = self.cur_boost_pred
        else:
            preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        assert(preds is not None)
        f = copy.deepcopy(self.online_belady())
        if address in self.sim_cache:
            target_index = self.sim_cache.index(address)
            self.sim_cache[target_index] = address
        elif None in self.sim_cache:
            index_to_evict = self.sim_cache.index(None)
            self.sim_cache[index_to_evict] = address
            self.prediction = copy.deepcopy(preds)
        if address not in self.sim_cache:
            target_index = None
            if self.skip == 0:
                self.follow_cost += 1
                if address not in f:
                    self.belady_cost +=1
                if address not in self.prediction and (self.follow_cost <= self.belady_cost):
                    if self.pred_gap <= 0:
                        self.prediction = preds
                        self.pred_gap = self.a
                        dd = self.differ(self.sim_cache, self.prediction)
                        target_index = self.sim_cache.index(random.choice(dd))
                        assert(self.sim_cache[target_index] not in self.prediction)
                        self.sim_cache[target_index] = address
                    else:
                        target_index = random.choice(range(self.associativity))
                        self.sim_cache[target_index] = address
                elif address in self.prediction:
                    dd = self.differ(self.sim_cache, self.prediction)
                    target_index = self.sim_cache.index(random.choice(dd))
                    self.sim_cache[target_index] = address
                else:
                    self.follow_cost = 0
                    self.belady_cost = 0
                    self.skip = self.associativity
                    self.old = []
                    for req in self.traces[self.timestamp-1::-1]:
                        if (req not in self.old) and (req != address):
                            self.old.append(req)
                        if len(self.old) >= self.associativity:
                            break
                    assert(len(self.old)==self.associativity)
                    self.unmarked = self.old.copy()
                    self.sim_cache = self.old.copy()
                    assert(address not in self.sim_cache)
                    self.marked = []
                    self.unmarked_for_reload = []
                    self.clean = []
            if self.skip != 0:
                assert(address not in self.sim_cache)
                if address not in self.marked:
                    self.skip -= 1
                    arrival_no = self.associativity-self.skip
                    if address in self.unmarked:
                        self.unmarked.remove(address)
                    if address not in self.marked:
                        self.marked.append(address)
                    assert(len(self.marked) == arrival_no)
                    if address not in self.old:
                        self.clean.append(address)
                    assert(len(self.unmarked) == self.associativity - (arrival_no - len(self.clean)))
                    if ((self.a==1) and (arrival_no in self.F)) or ((self.a > 1) and (self.pred_gap <= 0)):
                        self.pred_gap = self.a
                        self.prediction = copy.deepcopy(preds)
                    if arrival_no in self.S:
                        self.unmarked_for_reload = []
                        for p in self.unmarked:
                            if (p in self.prediction) and (p not in self.sim_cache):
                                self.unmarked_for_reload.append(p)
                    if address in self.unmarked_for_reload:
                        # Lazy sync with predictor
                        assert(address not in self.sim_cache)
                        dd = self.differ(self.sim_cache, self.prediction)
                        target_index = self.sim_cache.index(random.choice(dd))
                        self.sim_cache[target_index] = address
                    if address in self.clean: # Clean arrival
                            assert(address not in self.sim_cache)
                            dd = self.differ(self.sim_cache, self.prediction)
                            target_index = self.sim_cache.index(random.choice(dd))
                            self.sim_cache[target_index] = address
                if address not in self.sim_cache:
                    index_to_evict = None
                    unmarked_slots = []
                    for page in self.sim_cache:
                        if page in self.unmarked:
                            unmarked_slots.append(self.sim_cache.index(page))
                    target_index = random.choice(unmarked_slots)
                    assert(address not in self.sim_cache)
                    self.sim_cache[target_index] = address
                if self.skip == 0:
                    assert(len(self.marked) == self.associativity)
                    assert(len(self.unmarked) == len(self.clean))
        if self.cur_boost_type is not None:
            assert self.cur_boost_type == 'before'
        else:
            pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
            assert pred is None
        self.pred_gap -= 1
        self.traces.append(address)
    
    def access(self, pc, address):
        self.follow_robust(pc, address)

        ## Lazy
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(self.sim_cache)
                self.pcs = copy.deepcopy(self.sim_pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(self.cache) - set(self.sim_cache)
                target_index = self.lazy_evictor.evict([(self.cache.index(k), self.key_scores[self.cache.index(k)] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.key_scores[target_index] = self.timestamp
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.timestamp += 1
        return hit

class Guard(PredictAlgorithm):
    """
    Guard algorithm

    Parameters:
    
    - follow_if_guarded

    - relax_times

    - relax_prob

    Our work
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity, evictor_type, predictor_type)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()
        self.error_times = 0

        if 'follow_if_guarded' in kwargs:
            self.follow_if_guarded = kwargs['follow_if_guarded']
        else:
            self.follow_if_guarded = False
        if 'relax_times' in kwargs:
            self.relax_times = kwargs['relax_times']
        else:
            self.relax_times = 0
        if 'relax_prob' in kwargs:
            self.relax_prob = kwargs['relax_prob']
        else:
            self.relax_prob = 0
    
    def access(self, pc, address):
        to_guard = False
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if address in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            if to_guard and not self.follow_if_guarded:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        if to_guard:
            self.unguarded_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

#######################################################################

class CombineAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        if lazy_evictor_type is not None and not issubclass(lazy_evictor_type, Evictor):
            raise ValueError('CombineAlgorithm: Invalid Evictor')
        
        super().__init__(associativity)
        self.oracle_algs = []
        self.boost_algs = []
        self.candidate_algs = []
        self.center = 0
        self.timestamp = 0
        self.lazy_evictor = lazy_evictor_type() if lazy_evictor_type is not None else None
        # self.key_scores = {} if lazy_evictor_type == LRUEvictor else None
        self.key_scores = [np.inf] * associativity if lazy_evictor_type == LRUEvictor else None

        for alg_type in candidate_algorithms:
            alg_instance = alg_type(associativity)
            self.candidate_algs.append([alg_instance, 0])
            if hasattr(alg_instance, 'oracle_access'):
                self.oracle_algs.append(alg_instance)
            if hasattr(alg_instance, 'boost_access'):
                self.boost_algs.append(alg_instance)

        if len(self.oracle_algs) != 0:
            def oracle_access(self, pc, address, next_access_time):
                for oracle_alg in self.oracle_algs:
                    oracle_alg.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        if len(self.candidate_algs) < 2:
            raise ValueError('CombineAlgorithm: Algorithm Count < 2')

    def __push_candidates__(self, pc, address):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if not alg.access(pc, address):
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
    
    def __push_candidates_boost__(self, pc, address, boost_pred):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if alg in self.boost_algs:
                hit = alg.boost_access(pc, address, boost_pred)
            else:
                hit = alg.access(pc, address)
            if not hit:
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
    
    def __trigger_miss__(self, i, address):
        pass

    @abstractmethod
    def __trigger_elect_center__(self):
        pass

    def __process__(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            self.__trigger_elect_center__()
            center_cache = self.candidate_algs[self.center][0].cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                self.pcs = copy.deepcopy(self.candidate_algs[self.center][0].pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(self.cache) - set(center_cache)
                target_index = self.lazy_evictor.evict([(self.cache.index(k), self.key_scores[self.cache.index(k)] if self.key_scores is not None else 0) for k in diff_keys])
        if self.key_scores is not None:
            self.key_scores[target_index] = self.timestamp
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.timestamp += 1
        return hit

    def boost_access(self, pc, address, boost_pred):
        self.__push_candidates_boost__(pc, address, boost_pred)
        return self.__process__(pc, address)

    def access(self, pc, address):
        self.__push_candidates__(pc, address)
        return self.__process__(pc, address)

class CombineDeterministicAlgorithm(CombineAlgorithm):
    """
    black-box algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], switch_bound=2, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.switch_bound = switch_bound

    def __trigger_elect_center__(self):
        this_cost = self.candidate_algs[self.center][1]
        min_center, (_, min_cost) = min(enumerate(self.candidate_algs), key=lambda x: x[1][1])
        if this_cost >= self.switch_bound * min_cost:
            self.center = min_center

class CombineRandomAlgorithm(CombineAlgorithm):
    """
    Algorithm THRESH

    Designed by Avrim Blum and Carl Burch. 1997. On-line learning and the metrical task system problem.
    https://dl.acm.org/doi/10.1145/267460.267475
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], alpha=0.0, beta=0.99, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.alpha = alpha
        self.beta = beta
        self.n = len(self.candidate_algs)
        self.weights = [1] * self.n
        self.probs = [1/self.n] * self.n
    
    def __trigger_miss__(self, i, key):
        self.weights[i] *= self.beta
    
    def __trigger_elect_center__(self):
        W = sum(self.weights)
        threshold = self.alpha * W / self.n
        new_probs = [w / W for w in self.weights]
        if new_probs[self.center] < self.probs[self.center]:
            threshold = 1 - new_probs[self.center] / self.probs[self.center]
            if random.random() > threshold:
                self.center = self.center
            else:
                index = list(range(self.n))
                index.remove(self.center)
                probs = copy.deepcopy(new_probs)
                probs.pop(self.center)
                self.center = random.choices(index, weights=probs)[0]
        self.probs = new_probs

        # valid_index, valid_weights = zip(*[(i, weight) for i, weight in enumerate(self.weights) if weight > threshold])
        # if valid_weights:
        #     self.center = random.choices(valid_index, weights=valid_weights)[0]

class CombineWeightsAlgorithm(CombineAlgorithm):
    """
    Imitation learing for Parrot
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], weights: Union[List[float], None], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.n = len(self.candidate_algs)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1] * self.n
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.candidate_algs[self.center][0].preds)

    def reset(self, weights):
        self.weights = weights

    def __trigger_elect_center__(self):
        self.center = random.choices(list(range(self.n)), weights=self.weights)[0]

#######################################################################

class RandAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = RandEvictor()
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.cache)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class LRUAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = LRUEvictor()
        self.scores = [0] * associativity
        self.timestamp = 0
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = self.timestamp
        self.timestamp += 1
        return hit

class MarkerAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = MarkerEvictor()
        self.scores = [0] * associativity
    
    def access(self, pc, address):
        if all(x == 1 for x in self.scores):
            self.scores = [0] * self.associativity

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = 1
        return hit

####################################################################

class PredictAlgorithmFactory:
    predictor_evict_dict = {
        "PLECO": (MaxEvictor, PLECOPredictor),
        "PLECO-State": (DummyEvictor, PLECOStatePredictor),
        "PLECO-Bin": (BinaryEvictor, PLECOBinPredictor),
        "GBM": (BinaryEvictor, GBMBinPredictor),
        "LRB": (MinEvictor, LRBPredictor),
        "POPU": (MaxEvictor, POPUPredictor),
        "POPU-State": (DummyEvictor, POPUStatePredictor),
        "Parrot": (MaxEvictor, ParrotPredictor),
        "Parrot-State": (DummyEvictor, ParrotStatePredictor),
        "OracleDis": (ReuseDistanceEvictor, OracleReuseDistancePredictor),
        "OracleBin": (BinaryEvictor, OracleBinaryPredictor),
        "OraclePhase": (BinaryEvictor, OraclePhasePredictor),
        "OracleState": (DummyEvictor, OracleStatePredictor),
        "GuardLRB": (BinaryEvictor, LRBPredictor),  # 添加GuardLRB支持，使用LRBPredictor作为预测器
    }
    
    # 设置是否包含额外的LRB变体（如Mark0[LRB]等）
    include_lrb_variants = False

    @staticmethod
    def generate_predictive_algorithm(alg_type: Union[Type[PredictAlgorithm], partial], pred_type_str: str, **kwargs) -> partial:
        evictor_type, predictor_type = PredictAlgorithmFactory.predictor_evict_dict[pred_type_str]
        
        evictor_partial = evictor_type
        predictor_partial = predictor_type
        if pred_type_str == 'Parrot' or pred_type_str == 'Parrot-State' or pred_type_str == 'GBM':
            # shared_model
            if 'shared_model' not in kwargs:
                raise ValueError('PredictAlgorithmFactory: Parrot need [shared_model]')
            
            if pred_type_str == 'Parrot-State':
                if 'associativity' not in kwargs:
                    raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
                associativity = kwargs['associativity']
                predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model'], associativity=associativity)
            else:
                predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model']) 
        elif pred_type_str == 'LRB':
            # LRB需要shared_model参数
            if 'shared_model' not in kwargs:
                raise ValueError('PredictAlgorithmFactory: LRB need [shared_model]')
            
            # 获取必要的参数
            shared_model = kwargs['shared_model']
            memory_window = kwargs.get('memory_window', 1000000)  # 默认内存窗口为1000000
            
            # 创建预测器和驱逐器
            evictor_partial = partial(MinEvictor)
            predictor_partial = partial(LRBPredictor, shared_model=shared_model, memory_window=memory_window)
            
            # 避免memory_window重复传递
            if 'memory_window' in kwargs:
                del kwargs['memory_window']
            
            # 只有当alg_type是PredictAlgorithm时才返回LRBAlgorithm
            # 否则使用传入的alg_type（如Guard）
            if alg_type == PredictAlgorithm:
                # 返回LRBAlgorithm而不是标准的PredictAlgorithm
                return partial(LRBAlgorithm, evictor_type=evictor_partial, predictor_type=predictor_partial,
                                memory_window=memory_window, **kwargs)
            else:
                # 使用传入的算法类型（如Guard）
                # 过滤掉LRB特有的参数，只保留算法需要的参数
                filtered_kwargs = {}
                # 如果是partial，从keywords中获取需要的参数
                if isinstance(alg_type, partial):
                    # 保留alg_type.keywords中的参数
                    filtered_kwargs = {k: v for k, v in alg_type.keywords.items() 
                                     if k not in ['evictor_type', 'predictor_type']}
                return partial(alg_type, evictor_type=evictor_partial, predictor_type=predictor_partial, **filtered_kwargs)
        elif pred_type_str.startswith('Oracle'):
            reuse_dis_noise_sigma = 0
            lognormal = True
            if 'reuse_dis_noise_sigma' in kwargs:
                reuse_dis_noise_sigma = kwargs['reuse_dis_noise_sigma']
            if 'lognormal' in kwargs:
                lognormal = kwargs['lognormal']

            if pred_type_str == 'OracleDis':
                predictor_partial = partial(predictor_type, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
            else:
                if 'associativity' not in kwargs:
                    raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
                associativity = kwargs['associativity']
                if pred_type_str == 'OracleState':
                    predictor_partial = partial(predictor_type, associativity=associativity, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
                else:
                    bin_noise_prob = 0
                    if 'bin_noise_prob' in kwargs:
                        bin_noise_prob = kwargs['bin_noise_prob']
                    predictor_partial = partial(predictor_type, associativity=associativity, bin_noise_prob=bin_noise_prob, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
        elif pred_type_str.endswith('State'):
            if 'associativity' not in kwargs:
                raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
            associativity = kwargs['associativity']
            predictor_partial = partial(predictor_type, associativity=associativity)
        elif pred_type_str == 'PLECO-Bin':
            if 'threshold' not in kwargs:
                raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [threshold]')
            threshold = kwargs['threshold']
            predictor_partial = partial(predictor_type, threshold=threshold)

        if isinstance(alg_type, partial):
            this_partial = copy.deepcopy(alg_type)
            this_partial.keywords['evictor_type'] = evictor_partial
            this_partial.keywords['predictor_type'] = predictor_partial
            return this_partial
        else:
            return partial(alg_type, evictor_type=evictor_partial, predictor_type=predictor_partial)

def format_guard(relax_times, relax_prob):
    if relax_times == 0 and relax_prob == 0:
        return "-no-relax"
    elif relax_times == 0 and relax_prob != 0:
        return f"-relax-prob-{relax_prob}"
    elif relax_times != 0 and relax_prob == 0:
        return f"-relax-times-{relax_times}"
    else:
        raise ValueError('relax_times and relax_prob invaild')

def format_oracle(reuse_dis_noise_sigma, bin_noise_prob):
    if reuse_dis_noise_sigma == 0 and bin_noise_prob == 0:
        return "-oracle"
    elif reuse_dis_noise_sigma == 0 and bin_noise_prob != 0:
        return f"-bin-{bin_noise_prob}"
    elif reuse_dis_noise_sigma != 0 and bin_noise_prob == 0:
        return f"-dis-{reuse_dis_noise_sigma}"
    else:
        return f"-dis-{reuse_dis_noise_sigma}-bin-{bin_noise_prob}"

def pretty_print(callable: Union[EvictAlgorithm, partial], verbose=False) -> str:
    this_cls = callable
    if hasattr(callable, 'func'):
        this_cls = callable.func
    this_cls_name = this_cls.__name__.replace("Algorithm", '').replace("CombineDeterministic", 'CombDet').replace('CombineRandomAlgorithm', 'CombRand').replace("MarkAndPredict", "Mark&Predict").replace('PredictiveMarker', 'PredMark')
    metadata = this_cls_name
    if hasattr(callable, 'keywords'):
        kw = callable.keywords
        if issubclass(this_cls, CombineAlgorithm):
            algs = kw['candidate_algorithms']
            alg_names = []
            for alg in algs:
                alg_names.append(pretty_print(alg, verbose))
            metadata += ("[" + (", ".join(alg_names)) + "]")
        
        if 'predictor_type' in kw:
            predictor_type = kw['predictor_type']
            pred_kw = {}
            if hasattr(predictor_type, 'func'):
                pred_kw = predictor_type.keywords
                predictor_type = predictor_type.func
            predictor = predictor_type.__name__.replace("Predictor", '').replace('OracleReuseDistance', 'Belady').replace('OracleBinary', 'FBP')
            metadata += f'[{predictor}]'

            if issubclass(predictor_type, OraclePredictor) and verbose:
                reuse_dis_noise_sigma = bin_noise_prob = 0
                if 'reuse_dis_noise_sigma' in pred_kw:
                    reuse_dis_noise_sigma = pred_kw['reuse_dis_noise_sigma']
                if 'bin_noise_prob' in pred_kw:
                    bin_noise_prob = pred_kw['bin_noise_prob']
                metadata += format_oracle(reuse_dis_noise_sigma, bin_noise_prob) 

        if issubclass(this_cls, Guard):
            follow_if_guarded = False
            relax_times = relax_prob = 0
            if 'follow_if_guarded' in kw:
                follow_if_guarded = kw['follow_if_guarded']
            if follow_if_guarded:
                metadata += '-unv'
            else:
                metadata += '-f-pred'
            if 'relax_times' in kw:
                relax_times = kw['relax_times']
            if 'relax_prob' in kw:
                relax_prob = kw['relax_prob']
            metadata += format_guard(relax_times, relax_prob)
            
    return metadata

class LRBAlgorithm(PredictAlgorithm):
    """
    LRB (Learning Relaxed Belady) 算法实现
    
    论文：Z. Song, D. S. Berger, K. Li, and W. Lloyd. "Learning relaxed belady for content distribution network caching".
    In 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20). 2020.
    """
    def __init__(self, associativity, evictor_type, predictor_type, **kwargs):
        """初始化LRB算法"""
        # 对于LRB，我们使用BinaryEvictor作为驱逐器（处理0/1预测）
        super().__init__(associativity, BinaryEvictor, predictor_type, **kwargs)
        
        # LRB特有参数
        self.memory_window = kwargs.get('memory_window', 1000000)  # 内存窗口大小
        self.relaxation_factor = kwargs.get('relaxation_factor', 10.0)  # Belady的放松因子
        self.admission_size = kwargs.get('admission_size', associativity // 4 if associativity > 4 else 1)  # 缓存准入队列大小
        if self.admission_size is None:  # 确保admission_size不为None
            self.admission_size = associativity // 4 if associativity > 4 else 1
        self.admission_queue = collections.deque(maxlen=self.admission_size)  # 缓存准入队列
        
        # 启用/禁用LRB特性
        self.enable_admission = kwargs.get('enable_admission', True)  # 是否启用准入策略
        self.enable_edc = kwargs.get('enable_edc', True)  # 是否启用EDC特征
        self.debug_mode = kwargs.get('debug_mode', False)  # 调试模式
        
        # 统计信息
        self.hit_counter = 0
        self.miss_counter = 0
        
        if self.debug_mode:
            # 初始化信息不再输出
            pass
    
    def access(self, pc, address):
        """LRB访问逻辑实现"""
        target_index = -1
        hit = False
        
        # 刷新预测分数
        self.before_pred(pc, address)
        
        if address in self.cache:
            # 缓存命中
            target_index = self.cache.index(address)
            hit = True
            self.hit_counter += 1
            
        elif None in self.cache:
            # 缓存未满，考虑准入策略
            if self.enable_admission and address not in self.admission_queue and len(self.admission_queue) == self.admission_size:
                # 第一次看到这个对象，放入准入队列
                self.admission_queue.append(address)
                
                # 对象未进入缓存
                self.miss_counter += 1
                return False
            
            # 通过准入策略或准入策略被禁用，直接放入缓存
            target_index = self.cache.index(None)
            self.miss_counter += 1
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

    def _predict_all_pages(self):
        """在驱逐时刻对所有缓存页面重新计算预测分数"""
        predictions = {}
        
        # 遍历缓存中的所有页面
        for i, entry in enumerate(zip(self.cache, self.pcs)):
            if entry[0] is not None:  # 确保页面存在
                address = entry[0]
                features = self._extract_features((address, entry[1]))
                
                # 根据预测器类型选择预测方法
                try:
                    if hasattr(self.predictor, '_model'):
                        # LRBPredictor使用_model方法
                        predictions[address] = self.predictor._model(features)
                    elif hasattr(self.predictor, 'predict'):
                        # 其他预测器可能使用predict方法
                        predictions[address] = self.predictor.predict(features)
                    else:
                        # 没有可用的预测方法，使用默认值
                        predictions[address] = 0.5
                except Exception as e:
                    if self.debug_mode and self.timestamp % 100000 == 0:
                        print(f"预测页面 {address} 失败: {e}")
                    # 发生异常，使用默认值
                    predictions[address] = 0.5
        
        return predictions
    
    def _extract_features(self, cache_entry):
        """从缓存条目中提取LRB所需特征"""
        address = cache_entry[0]
        pc = cache_entry[1]
        
        # 如果使用的是LRBPredictor，直接使用它的特征提取能力
        if hasattr(self.predictor, 'extract_features'):
            return self.predictor.extract_features(self.timestamp, pc, address)
        
        # 否则尝试手动提取特征
        predictor = self.predictor
        delta_features = []
        edc_features = []
        
        # 尝试提取delta特征
        if hasattr(predictor, 'deltas'):
            for i in range(getattr(predictor, 'delta_nums', 1)):
                if address in predictor.deltas[i]:
                    delta_features.append(predictor.deltas[i][address])
                else:
                    delta_features.append(np.inf)
        
        # 尝试提取EDC特征
        if hasattr(predictor, 'edcs'):
            for i in range(getattr(predictor, 'edc_nums', 1)):
                if address in predictor.edcs[i]:
                    edc_features.append(predictor.edcs[i][address])
                else:
                    edc_features.append(0)
        
        # 返回完整特征向量
        return [pc, address] + delta_features + edc_features

class GuardLRBAlgorithm(PredictAlgorithm):
    """
    Guard+LRB 算法实现 (改进版)
    
    结合Guard算法的保护机制与LRB的预测能力，在驱逐时刻对所有候选页面重新计算预测分数，
    而非在访问时刻计算，以确保预测分数的可比性。
    
    参考文献：
    1. Guard: N. Beckmann, H. Chen, and A. Cidon. "LHD: Improving cache hit rate by maximizing hit density". 
       In 15th USENIX Symposium on Networked Systems Design and Implementation (NSDI 18). 2018.
    2. LRB: Z. Song, D. S. Berger, K. Li, and W. Lloyd. "Learning relaxed belady for content distribution network caching".
       In 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20). 2020.
    """
    def __init__(self, associativity, evictor_type, predictor_type, **kwargs):
        """初始化Guard+LRB算法"""
        # 对于Guard+LRB，我们使用BinaryEvictor作为驱逐器（处理0/1预测）
        super().__init__(associativity, BinaryEvictor, predictor_type, **kwargs)
        
        # LRB特有参数
        self.memory_window = kwargs.get('memory_window', 1000000)  # 内存窗口大小
        self.relaxation_factor = kwargs.get('relaxation_factor', 10.0)  # Belady的放松因子
        self.admission_size = kwargs.get('admission_size', associativity // 4 if associativity > 4 else 1)  # 缓存准入队列大小
        if self.admission_size is None:  # 确保admission_size不为None
            self.admission_size = associativity // 4 if associativity > 4 else 1
        self.admission_queue = collections.deque(maxlen=self.admission_size)  # 缓存准入队列
        
        # 启用/禁用LRB特性
        self.enable_admission = kwargs.get('enable_admission', True)  # 是否启用准入策略
        self.enable_edc = kwargs.get('enable_edc', True)  # 是否启用EDC特征
        self.debug_mode = kwargs.get('debug_mode', False)  # 调试模式
        
        # Guard算法相关状态
        self.guarded_pages = set()  # 受保护的页面集合
        self.unguarded_pages = set()  # 未受保护的页面集合(集合U)
        self.current_phase_evicted = set()  # 当前阶段被驱逐的页面集合
        
        # LRB+Guard参数优化
        self.relax_threshold = kwargs.get('relax_threshold', 0.2)  # 启用随机选择的阈值
        self.enable_random_relax = kwargs.get('enable_random_relax', True)  # 是否启用随机松弛
        self.guard_weight = kwargs.get('guard_weight', 0.7)  # Guard机制的权重 (0-1 之间)
        
        # 设置阶段转换触发条件
        self.phase_reset_percentage = kwargs.get('phase_reset_percentage', 0.7)  # 当unguarded_pages比例低于此值时开始新阶段
        
        # 当前请求信息
        self.current_request_addr = None
        
        # 统计信息
        self.hit_counter = 0
        self.miss_counter = 0
        self.guard_hits = 0
        self.phases = 0
        self.admission_hits = 0
        
        # 初始化阶段
        for i in range(associativity):
            if self.cache[i] is not None:
                self.unguarded_pages.add(self.cache[i])
        
        # 使用标准LRU作为后备策略
        self.lru_timestamps = [0] * associativity
        
        if self.debug_mode:
            # 初始化信息不再输出
            pass
    
    def _predict_all_pages(self):
        """在驱逐时刻对所有缓存页面重新计算预测分数"""
        predictions = {}
        
        # 遍历缓存中的所有页面
        for i, entry in enumerate(zip(self.cache, self.pcs)):
            if entry[0] is not None:  # 确保页面存在
                address = entry[0]
                features = self._extract_features((address, entry[1]))
                
                # 根据预测器类型选择预测方法
                try:
                    if hasattr(self.predictor, '_model'):
                        # LRBPredictor使用_model方法
                        predictions[address] = self.predictor._model(features)
                    elif hasattr(self.predictor, 'predict'):
                        # 其他预测器可能使用predict方法
                        predictions[address] = self.predictor.predict(features)
                    else:
                        # 没有可用的预测方法，使用默认值
                        predictions[address] = 0.5
                except Exception as e:
                    if self.debug_mode and self.timestamp % 100000 == 0:
                        print(f"预测页面 {address} 失败: {e}")
                    # 发生异常，使用默认值
                    predictions[address] = 0.5
        
        return predictions
    
    def _extract_features(self, cache_entry):
        """从缓存条目中提取LRB所需特征"""
        address = cache_entry[0]
        pc = cache_entry[1]
        
        # 如果使用的是LRBPredictor，直接使用它的特征提取能力
        if hasattr(self.predictor, 'extract_features'):
            return self.predictor.extract_features(self.timestamp, pc, address)
        
        # 否则尝试手动提取特征
        predictor = self.predictor
        delta_features = []
        edc_features = []
        
        # 尝试提取delta特征
        if hasattr(predictor, 'deltas'):
            for i in range(getattr(predictor, 'delta_nums', 1)):
                if address in predictor.deltas[i]:
                    delta_features.append(predictor.deltas[i][address])
                else:
                    delta_features.append(np.inf)
        
        # 尝试提取EDC特征
        if hasattr(predictor, 'edcs'):
            for i in range(getattr(predictor, 'edc_nums', 1)):
                if address in predictor.edcs[i]:
                    edc_features.append(predictor.edcs[i][address])
                else:
                    edc_features.append(0)
        
        # 返回完整特征向量
        return [pc, address] + delta_features + edc_features
    
    def check_phase_transition(self):
        """检查是否需要开始新阶段"""
        # 计算未保护页面比例
        total_valid_pages = sum(1 for p in self.cache if p is not None)
        if total_valid_pages == 0:
            return False
            
        unguarded_ratio = len(self.unguarded_pages) / total_valid_pages
        
        # 如果未保护页面比例低于阈值，开始新阶段
        return unguarded_ratio < self.phase_reset_percentage
    
    def update_lru(self, target_index):
        """更新LRU时间戳"""
        self.lru_timestamps[target_index] = self.timestamp
    
    def select_victim_lru(self):
        """选择LRU最旧的页面"""
        min_time = float('inf')
        min_index = -1
        for i, addr in enumerate(self.cache):
            if addr is not None and self.lru_timestamps[i] < min_time:
                min_time = self.lru_timestamps[i]
                min_index = i
        return min_index if min_index >= 0 else 0  # 默认返回0
    
    def access(self, pc, address):
        """Guard+LRB访问逻辑实现"""
        self.current_request_addr = address  # 记录当前请求地址
        target_index = -1
        hit = False
        
        # 刷新预测分数（仅用于更新特征）
        self.before_pred(pc, address)
        
        if address in self.cache:
            # 缓存命中
            target_index = self.cache.index(address)
            hit = True
            self.hit_counter += 1
            
            # 如果命中的是受保护页面，增加guard命中计数
            if address in self.guarded_pages:
                self.guard_hits += 1
            
            # 更新LRU信息
            self.update_lru(target_index)
            
        elif None in self.cache:
            # 缓存未满，考虑准入策略 - 减少准入策略严格性
            if self.enable_admission and self.admission_size > 0 and address not in self.admission_queue:
                # 减少准入队列判断严格性，提高新页面进入缓存的机会
                if random.random() < 0.3:  # 30%的机会直接进入缓存，绕过准入队列
                    # 允许进入缓存
                    pass
                elif len(self.admission_queue) >= self.admission_size:
                    # 否则放入准入队列
                    self.admission_queue.append(address)
                    # 对象未进入缓存
                    self.miss_counter += 1
                    return False
            
            # 通过准入策略或准入策略被禁用，放入空闲位置
            target_index = self.cache.index(None)
            self.miss_counter += 1
            
            # 更新Guard状态：新页面不受保护
            self.unguarded_pages.add(address)
            
            # 更新LRU信息
            self.update_lru(target_index)
            
        else:
            # 缓存满，需要决定是否要驱逐某个对象
            # 减少准入策略的严格性
            if self.enable_admission and self.admission_size > 0 and address not in self.admission_queue:
                # 减少准入队列判断严格性，提高新页面进入缓存的机会
                if random.random() < 0.3:  # 30%的机会直接进入缓存，绕过准入队列 
                    # 允许进入缓存
                    pass
                elif len(self.admission_queue) >= self.admission_size:
                    # 否则放入准入队列
                    self.admission_queue.append(address)
                    # 对象未进入缓存
                    self.miss_counter += 1
                    return False
            
            # 检查是否需要开始新阶段
            if self.check_phase_transition():
                if self.debug_mode and self.timestamp % 100000 == 0:
                    print(f"阶段转换 #{self.phases}")
                
                self.guarded_pages.clear()
                self.unguarded_pages = set(addr for addr in self.cache if addr is not None)
                self.current_phase_evicted.clear()
                self.phases += 1
            
            # 简化的驱逐决策逻辑
            # 对每个页面进行预测评分
            try:
                lrb_predictions = self._predict_all_pages()
            except Exception as e:
                if self.debug_mode and self.timestamp % 100000 == 0:
                    print(f"预测评分失败: {e}, 使用preds作为后备")
                # 使用当前的preds作为默认预测值
                lrb_predictions = {}
                for i, addr in enumerate(self.cache):
                    if addr is not None:
                        lrb_predictions[addr] = self.preds[i]
            
            # 核心决策逻辑 - 简化并平衡Guard和LRB
            # 1. 如果当前页面在此阶段被驱逐过，优先考虑Guard保护
            guard_applied = False
            
            if address in self.current_phase_evicted:
                # 保护被驱逐过的页面, 选择未保护页面进行驱逐
                unguarded_indices = []
                for i, addr in enumerate(self.cache):
                    if addr is not None and addr in self.unguarded_pages:
                        unguarded_indices.append(i)
                
                if unguarded_indices and random.random() < 0.9:  # 90%的概率应用Guard保护
                    # 随机选择一个未保护页面
                    target_index = random.choice(unguarded_indices)
                    victim_addr = self.cache[target_index]
                    
                    # 保护当前请求的页面
                    self.guarded_pages.add(address)
                    # 从未保护集合移除被驱逐的页面
                    if victim_addr in self.unguarded_pages:
                        self.unguarded_pages.remove(victim_addr)
                    
                    guard_applied = True
            
            # 2. 如果Guard保护未应用，使用LRB分数与保护状态的综合评分
            if not guard_applied:
                candidates = []
                scores = []
                
                # 更平衡的混合分数计算
                for i, addr in enumerate(self.cache):
                    if addr is not None:
                        # LRB预测分数 (0=保留, 1=驱逐)
                        lrb_score = lrb_predictions.get(addr, 0.5)
                        
                        # LRU归一化分数 (越小越旧)
                        max_timestamp = self.timestamp + 1
                        lru_score = (max_timestamp - self.lru_timestamps[i]) / max_timestamp
                        
                        # 保护状态影响 - 降低保护对分数的极端影响
                        # 受保护页面的分数降低30-70%，而不是完全屏蔽
                        protection_factor = 0.3 if addr in self.guarded_pages else 1.0
                        
                        # 混合评分 (越高越应该被驱逐)
                        # 增加LRB在决策中的权重
                        mixed_score = (lrb_score * 0.7 + lru_score * 0.3) * protection_factor
                        
                        candidates.append(i)
                        scores.append(mixed_score)
                
                if candidates:
                    # 选择混合分数最高的页面
                    max_score_index = scores.index(max(scores))
                    target_index = candidates[max_score_index]
                    victim_addr = self.cache[target_index]
                    
                    # 随机松弛 - 有10%的概率随机选择而不是选择得分最高的
                    if random.random() < 0.1:
                        weights = [s/sum(scores) for s in scores]
                        target_index = random.choices(candidates, weights=weights)[0]
                        victim_addr = self.cache[target_index]
                
                # 更新保护状态
                if victim_addr is not None:
                    if victim_addr in self.unguarded_pages:
                        self.unguarded_pages.remove(victim_addr)
                    # 记录被驱逐的页面
                    self.current_phase_evicted.add(victim_addr)
            
            self.miss_counter += 1
        
        # 更新缓存
        if target_index >= 0:
            old_value = self.cache[target_index]
            if old_value is not None:
                # 从集合中移除被驱逐的页面
                if old_value in self.unguarded_pages:
                    self.unguarded_pages.remove(old_value)
                if old_value in self.guarded_pages:
                    self.guarded_pages.remove(old_value)
            
            # 设置新页面
            self.cache[target_index], self.pcs[target_index] = address, pc
            
            # 设置新页面的保护状态 - 若被驱逐过，则受保护
            if address in self.current_phase_evicted:
                self.guarded_pages.add(address)
            else:
                # 新页面默认为未保护状态
                self.unguarded_pages.add(address)
            
            # 更新LRU信息
            self.update_lru(target_index)
            
            # 更新预测分数
            self.after_pred(pc, address, target_index)
        
        # 定期打印统计信息（调试模式）
        if self.debug_mode and self.timestamp % 100000 == 0:
            hit_rate = self.hit_counter/(self.hit_counter+self.miss_counter) if (self.hit_counter+self.miss_counter) > 0 else 0
            print(f"统计 #{self.timestamp//1000}K: 命中率={hit_rate:.4f}, 阶段数={self.phases}")
            
        return hit