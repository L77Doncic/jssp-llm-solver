"""Instance 数据结构、结构校验与序列化测试。"""

import pytest

from problem.instance import Instance, from_raw, validate_structure


def make_valid(n=3, m=3):
    machines = [[(j + k) % m for k in range(m)] for j in range(n)]
    durations = [[1] * m for _ in range(n)]
    return Instance(instance_id="test-00000", n=n, m=m, machines=machines, durations=durations)


class TestStructure:
    def test_valid_instance_ok(self):
        make_valid()

    def test_empty_instance_id_rejected(self):
        with pytest.raises(ValueError, match="instance_id"):
            Instance(instance_id="", n=1, m=1, machines=[[0]], durations=[[1]])

    def test_machine_out_of_range(self):
        with pytest.raises(ValueError, match="超出"):
            Instance(instance_id="x", n=1, m=2, machines=[[5, 0]], durations=[[1, 1]])

    def test_nonpositive_duration(self):
        with pytest.raises(ValueError, match="正整数"):
            Instance(instance_id="x", n=1, m=1, machines=[[0]], durations=[[0]])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            Instance(instance_id="x", n=2, m=2, machines=[[0, 1]], durations=[[1, 1], [1, 1]])

    def test_validate_structure_returns_errors(self):
        errors = validate_structure(1, 2, [[0, 9]], [[1, -3]])
        assert len(errors) == 2  # 机器越界 + 非正时间


class TestSerialization:
    def test_dict_roundtrip(self):
        inst = make_valid()
        assert Instance.from_dict(inst.to_dict()) == inst

    def test_json_roundtrip(self):
        inst = make_valid()
        assert Instance.from_json(inst.to_json()) == inst

    def test_from_dict_missing_field(self):
        with pytest.raises(ValueError, match="缺少字段"):
            Instance.from_dict({"n": 1})

    def test_from_json_bad_json(self):
        with pytest.raises(Exception):
            Instance.from_json("not json")


class TestFromRaw:
    def test_zero_based_offset(self):
        inst = from_raw("raw-0", 1, 2, [[(0, 5), (1, 7)]], machine_offset=0)
        assert inst.machines == [[0, 1]]
        assert inst.durations == [[5, 7]]

    def test_one_based_offset(self):
        # OR-Library la 系列惯例：机器编号从 1 开始
        inst = from_raw("raw-1", 2, 2, [[(1, 5), (2, 7)], [(2, 3), (1, 4)]], machine_offset=1)
        assert inst.machines == [[0, 1], [1, 0]]
        assert inst.durations == [[5, 7], [3, 4]]

    def test_from_raw_shape_check(self):
        with pytest.raises(ValueError):
            from_raw("raw-1", 2, 2, [[(1, 5)]], machine_offset=0)

    def test_to_text_contains_jobs(self):
        text = make_valid().to_text()
        assert "3 jobs x 3 machines" in text
        assert "Job 0" in text and "Job 2" in text
