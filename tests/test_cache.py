"""Kiểm thử StageCache và SegParams.cached (FR-94)."""
from backend.pipeline.cache import StageCache
from backend.pipeline.schema import SegParams


def _counter():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return object()
    return calls, build


def test_get_or_builds_once_per_key():
    cache = StageCache()
    calls, build = _counter()
    a = cache.get_or(("k", 0), build)
    b = cache.get_or(("k", 0), build)
    assert calls["n"] == 1 and a is b


def test_different_keys_build_separately():
    cache = StageCache()
    calls, build = _counter()
    cache.get_or(("k", 0), build)
    cache.get_or(("k", 1), build)
    assert calls["n"] == 2


def test_manual_box_person_key_differs_from_person_0():
    cache = StageCache()
    b1, b2 = object(), object()
    assert cache.get_or(("gcmask", -1), lambda: b1) is not cache.get_or(("gcmask", 0), lambda: b2)


def test_same_idx_different_box_are_different_entries():
    cache = StageCache()
    k1, k2 = ("roi", -1, (0, 0, 10, 10)), ("roi", -1, (5, 5, 20, 20))
    a = cache.get_or(k1, lambda: object())
    b = cache.get_or(k2, lambda: object())
    assert a is not b and len(cache) == 2
    assert cache.get_or(k1, lambda: object()) is a


def test_lru_evicts_oldest_beyond_maxsize():
    cache = StageCache(maxsize=2)
    for i in range(3):
        cache.get_or(("k", i), lambda: i)
    assert len(cache) == 2
    calls, build = _counter()
    cache.get_or(("k", 0), build)   # đã bị đẩy ra -> dựng lại
    assert calls["n"] == 1


def test_clear_empties_cache():
    cache = StageCache()
    cache.get_or(("k",), lambda: 1)
    cache.clear()
    assert len(cache) == 0


def test_segparams_cached_works_without_cache():
    assert SegParams().cached(("k",), lambda: 7) == 7


def test_segparams_cached_uses_cache():
    calls, build = _counter()
    p = SegParams(cache=StageCache())
    assert p.cached(("k",), build) is p.cached(("k",), build)
    assert calls["n"] == 1
