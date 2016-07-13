"""Test hashdict

The hashdict is a very simple structure which inherits almost all of its
properties from the python standard dict. Tests are here mostly to
demonstrate the interface.
"""

import pickle
import metis.hashdict

def test_create_hashdict():
    """Check that a hashdict can be created from a dict"""
    hd1 = metis.hashdict.hashdict({'a': 1, 'b': 2})

    example = {'a': 1, 'b': 2}
    hd2 = metis.hashdict.hashdict(example)
    assert hash(hd1) == hash(hd2)

def test_hashdict_not_mutable():
    """Verify hashdict can't be modified"""
    hd1 = metis.hashdict.hashdict({'a': 1, 'b': 2})
    def hashdict_set():
        """Stub to check that __setitem__ raises an error"""
        hd1['a'] = 2
    def hashdict_del():
        """Stub to check that __delitem__ raises an error"""
        del hd1['a']
    for name, function in {
            'set': hashdict_set,
            'del': hashdict_del,
            'clear': lambda: hd1.clear(), # pylint: disable=unnecessary-lambda
            'pop': lambda: hd1.pop('a'),
            'popitem': lambda: hd1.popitem(('a', 1)),
            'setdefault': lambda: hd1.setdefault('a'),
            'update': lambda: hd1.update({'c': 3})
        }.iteritems():
        yield check_raises_typeerror, name, function

def check_raises_typeerror(name, func):
    """Check that 'func' raises TypeError when called"""
    try:
        func()
    except TypeError:
        pass
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "Mutating method '{0}' is not guarded by an exception".format(name))


def test_pickleable():
    hd1 = metis.hashdict.hashdict({'a': 1, 'b': 2})
    hd2 = pickle.loads(pickle.dumps(hd1))
    assert hd1 == hd2
