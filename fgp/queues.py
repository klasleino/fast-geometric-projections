try:
    from queue import Queue, PriorityQueue
except ImportError:
    from Queue import Queue, PriorityQueue

from itertools import count

from fgp.utils import set_hash


class DQueue:
    '''
    Queue data-structure allowing us to dequeue decision boundaries first.
    '''
    def __init__(self, visited, batch_size=1):
        self.decisions=Queue()
        self.activations=Queue()

        self.batch_size = batch_size
        self.visited = visited

    def get(self):
        constrs = []
        while not self.decisions.empty():
            constrs.append(self.decisions.get(block=False))

        if constrs:
            return constrs

        i = 0
        while not self.activations.empty() and i < self.batch_size:
            d, C, A, rep = self.activations.get(block=False)

            if set_hash(rep) not in self.visited:
                constrs.append((d, C, A, rep))

                self.visited.add(set_hash(rep))
                i += 1
        
        return constrs

    def empty(self):
        return self.decisions.empty() and self.activations.empty()

    def qsize(self):
        return self.decisions.qsize() + self.activations.qsize()

    def put(self, x):
        d, C, A, rep = x
        is_decision = rep is None

        if is_decision:
            self.decisions.put((d, C, A, rep))
        else:
            self.activations.put((d, C, A, rep))


class PQueue(object):
    '''
    Priority queue data-structure.
    '''
    def __init__(self, visited):
        self.queue = PriorityQueue()

        # Batch size must always be 1.
        self.batch_size = 1
        self.visited = visited

        self.unique = count()

    def get(self):
        constrs = []

        i = 0
        while not self.queue.empty() and i < self.batch_size:
            d, _, C, A, rep = self.queue.get(block=False)

            if rep is None or set_hash(rep) not in self.visited:
                constrs.append((d, C, A, rep))

                if rep is not None:
                    self.visited.add(set_hash(rep))
                i += 1
        
        return constrs

    def empty(self):
        return self.queue.empty()

    def qsize(self):
        return self.queue.qsize()

    def put(self, x):
        d, C, A, rep = x
        self.queue.put(d, next(self.unique), C, A, rep)
