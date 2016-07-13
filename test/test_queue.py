"""Test queue and related metis"""
import metis

def test_insert():
    """Test the insertion and containment methods"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2
    assert len(queue) == 3
    assert queue.top_key() == 0
    assert queue.top() == "zero"
    assert "zero" in queue
    assert "one" in queue
    assert "two" in queue

def test_pop():
    """Test the pop method"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2

    vertex = queue.pop()
    assert vertex == "zero"

    assert len(queue) == 2
    assert queue.top_key() == 1
    assert queue.top() == "one"

def test_remove():
    """Test the remove method"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2

    del queue['zero']
    assert len(queue) == 2
    assert queue.top_key() == 1
    assert queue.top() == "one"

def test_clear():
    """Test the ability to empty the queue"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2
    queue.clear()
    assert len(queue) == 0

def test_modify():
    """Test the in-place modification methods"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2

    queue["zero"] = 3
    queue["one"] = 3
    assert len(queue) == 3
    assert queue.top_key() == 2
    assert queue.top() == "two"

def test_sorted():
    """Test sorted iteration"""
    queue = metis.queue.PriorityQueue()
    queue["zero"] = 0
    queue["one"] = 1
    queue["two"] = 2
    sorted_queue = [i for i in queue.sorted()]
    assert all(sorted_queue[i] <= sorted_queue[i+1]
               for i in xrange(len(queue)-1))
