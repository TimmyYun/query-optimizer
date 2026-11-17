import pytest

from benchmark_utils import resolve_buckets


def test_resolve_buckets_forms():
    assert resolve_buckets("100", 873) == 100
    assert resolve_buckets("fd", 873) == 873
    assert resolve_buckets("FD", 873) == 873
    assert resolve_buckets("fd*1", 873) == 873
    assert resolve_buckets("fd*1.2", 873) == 1048
    assert resolve_buckets("fd*1.4", 873) == 1222
    assert resolve_buckets("fd*2", 873) == 1746
    assert resolve_buckets("fd*0.001", 10) == 1


def test_resolve_buckets_rejects_garbage():
    with pytest.raises(ValueError):
        resolve_buckets("fdx2", 873)
