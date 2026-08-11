"""实例生成器测试：合法性、经典性（permutation）、可复现性。"""

import pytest

from problem.generator import generate_batch, generate_instance
from problem.instance import Instance


class TestSingleGeneration:
    def test_permutation_machines(self):
        """经典 JSSP：每个工件的机器访问顺序是 0..m-1 的排列。"""
        inst = generate_instance(5, 7, seed=42, index=0)
        assert isinstance(inst, Instance)
        for j in range(inst.n):
            assert sorted(inst.machines[j]) == list(range(inst.m))
            assert all(1 <= p <= 99 for p in inst.durations[j])

    def test_duration_range(self):
        inst = generate_instance(3, 3, seed=0, index=0, p_min=10, p_max=20)
        assert all(10 <= p <= 20 for row in inst.durations for p in row)

    def test_id_format(self):
        inst = generate_instance(6, 6, seed=7, index=3)
        assert inst.instance_id == "gen-6x6-7-00003"

    def test_reproducible(self):
        a = generate_instance(4, 5, seed=99, index=2)
        b = generate_instance(4, 5, seed=99, index=2)
        assert a == b

    def test_different_seed_different_instance(self):
        a = generate_instance(4, 5, seed=1, index=0)
        b = generate_instance(4, 5, seed=2, index=0)
        assert a.machines != b.machines or a.durations != b.durations


class TestBatch:
    def test_count_and_ids_unique(self):
        batch = generate_batch(3, 3, count=10, seed=0)
        assert len(batch) == 10
        ids = [inst.instance_id for inst in batch]
        assert len(set(ids)) == 10
        assert ids[0] == "gen-3x3-0-00000"
        assert ids[9] == "gen-3x3-0-00009"

    def test_batch_reproducible(self):
        a = generate_batch(4, 4, count=5, seed=123)
        b = generate_batch(4, 4, count=5, seed=123)
        assert a == b

    def test_zero_count(self):
        assert generate_batch(3, 3, count=0, seed=0) == []

    def test_negative_count_rejected(self):
        with pytest.raises(ValueError):
            generate_batch(3, 3, count=-1, seed=0)
