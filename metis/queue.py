"""Priority queue implementation designed for planning algorithms"""

import heapq

class PriorityQueue(dict):
    """Priority queue implementation based on dict and heapq

    This priority queue is backed by a dict, and all dictionary methods
    work as expected. It combines the desirable feature of a heap (O(1)
    retrieval, O(log n) removal of the lowest priority item and O(log n)
    insertion of new items) with the desirable features of a dict (O(1)
    containment checking and priority retrieval). It additionally
    provides amortized O(log n) update of item priorities.

    For the purposes of the dict interface, the keys are the items in
    the queue (so items must be hashable) and the values are priorities
    (which must be comparable). Ties in priority are broken by comparing
    items, so items must also be comparable.

    This code is modified from a 2007 ActiveState recipe by Matteo
    Dell'Amico [1], itself an updated version of a 2002 ActiveState
    recipy from David Eppstein [2].

    Examples:
        >>> q = PriorityQueue([(1, "one"), (2, "two")])
        >>> q.top()
        'one'
        >>> q.top_key()
        1
        >>> q['zero'] = 0
        >>> q.top()
        'zero'
        >>> q.pop()
        'zero'
        >>> q.top()
        'one'
        >>> q['one'] = 3
        >>> q.top()
        'two'
        >>> 'two' in q
        True
        >>> len(q)
        2
        >>> q.clear()
        >>> len(q)
        0

    [1] http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/
    [2] http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228%29.?_ga=1.93537757.1561469608.1467776173
    """
    def __init__(self, *args, **kwargs):
        super(PriorityQueue, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapq.heapify(self._heap)

    def top_pair(self):
        """Return the (item, priority) pair with the lowest priority.

        The dictionary is the canonical representation of containment
        and priority; the heap is present only for efficeint retrieval
        and removal of the lowest priority item. Because the heap
        becomes desyncrhonized from the dictionary during insertion, we
        need to repeatedly pop items from the heap until we find an item
        which is in the dictionary.  This operation takes amortized O(1)
        time.

        Raises:
            IndexError if the object is empty.
        """
        heap = self._heap
        priority, item = heap[0]
        while item not in self or self[item] != priority:
            heapq.heappop(heap)
            priority, item = heap[0]
        return (item, priority)


    def top(self):
        """Return the item with the lowest priority.

        Returns:
            the item with the lowest priority

        Raises:
            IndexError if the object is empty.
        """
        return self.top_pair()[0]

    def top_key(self):
        """Return the item with the lowest priority.

        Returns:
            the item with the lowest priority

        Raises:
            IndexError if the object is empty.
        """
        return self.top_pair()[1]

    def pop(self):
        """Return the item with the lowest priority and remove it.

        Raises:
            IndexError if the object is empty.
        """
        heap = self._heap
        priority, item = heapq.heappop(heap)
        while item not in self or self[item] != priority:
            priority, item = heapq.heappop(heap)
        del self[item]
        return item

    def __setitem__(self, key, val):
        """Insert an item and set its priority

        Updating the priority of an item on the queue, or removing an
        item from the heap, takes O(n) time; we avoid this cost by
        leaving old items on the heap and inserting new ones.

        To keep from using too much space, we periodically reconstruct
        the heap. We do this whenever the heap grows more than twice the
        size of the dictionary: this operation occurs at most every
        log(n) insertions/updates, and takes O(n log n) time, so the
        operation takes amortized O(n log(n)) time.
        """
        super(PriorityQueue, self).__setitem__(key, val)
        if len(self._heap) < 2 * len(self):
            heapq.heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            # Delegate insertion to __setitem__
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        """Merge two priority queues

        Reimplementing dict.update is tricky: see e.g. [1]. Moreover,
        merging heaps takes O(n) time, and the heapq module does not
        provide an interface for merging heaps. So we just delegate the
        update to `dict` and rebuild the heap from scratch.

        [1] http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        """
        super(PriorityQueue, self).update(*args, **kwargs)
        self._rebuild_heap()

    def clear(self):
        """Remove all vertices from the queue"""
        super(PriorityQueue, self).clear()
        del self._heap[:]

    def sorted(self):
        """Iterate over the queue in sorted order"""
        self._heap.sort()
        return iter(self._heap)

    def __str__(self):
        return "Priority queue\n" + "\n".join("  {}: {}".format(
            item[0], item[1]) for item in self.sorted())

    def __repr__(self):
        return "{0}([{1}])".format(self.__class__.__name__, repr(self._heap))

