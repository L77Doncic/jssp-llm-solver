"""标准实例解析器测试（OR-Library / Taillard 两种惯例）。"""

from pathlib import Path

import pytest

from data.parsers import parse_jobshop_file


FT06_RAW = """6 6
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
"""

LA01_RAW = """10 5
2 44 3 5 5 58 4 97 1 9
3 15 1 31 2 87 4 100 5 8
2 11 5 16 3 31 4 76 1 78
5 24 4 50 1 30 2 60 3 51
4 34 2 11 5 15 3 45 1 35
1 26 5 40 2 55 4 43 3 78
2 30 5 25 1 48 3 79 4 36
5 34 1 44 2 41 4 82 3 25
4 74 3 88 5 42 2 39 1 74
2 24 1 42 3 36 5 62 4 24
"""

TA01_RAW = """10 10
0 43 1 89 2 4 3 32 4 9 5 8 6 28 7 69 8 64 9 57
1 57 2 16 3 42 4 83 5 42 6 7 7 69 8 63 9 40 0 35
2 55 3 28 4 20 5 29 6 25 7 79 8 28 9 15 0 36 1 34
3 28 4 61 5 36 6 43 7 55 8 49 9 62 0 44 1 20 2 30
4 27 5 94 6 60 7 40 8 48 9 69 0 9 1 25 2 43 3 38
5 14 6 41 7 29 8 95 9 58 0 50 1 27 2 32 3 35 4 17
6 49 7 73 8 32 9 54 0 32 1 30 2 69 3 74 4 25 5 30
7 78 8 55 9 12 0 61 1 54 2 65 3 64 4 79 5 26 6 31
8 22 9 41 0 25 1 30 2 73 3 67 4 59 5 64 6 18 7 14
9 14 0 71 1 61 2 47 3 63 4 49 5 80 6 63 7 18 8 29
"""


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "ft06.txt").write_text(FT06_RAW, encoding="utf-8")
    (tmp_path / "la01.txt").write_text(LA01_RAW, encoding="utf-8")
    (tmp_path / "ta01.txt").write_text(TA01_RAW, encoding="utf-8")
    return tmp_path


class TestParsing:
    def test_ft06_zero_based(self, raw_dir):
        inst = parse_jobshop_file(raw_dir / "ft06.txt")
        assert inst.instance_id == "ft06"
        assert (inst.n, inst.m) == (6, 6)
        # 0-based：首工件首工序机器号 2
        assert inst.machines[0][0] == 2
        assert inst.durations[0][0] == 1
        # 每工件是 permutation
        for j in range(6):
            assert sorted(inst.machines[j]) == list(range(6))

    def test_la01_one_based(self, raw_dir):
        inst = parse_jobshop_file(raw_dir / "la01.txt")
        assert (inst.n, inst.m) == (10, 5)
        # 1-based 归一化：首工件首工序机器号 2 → 1
        assert inst.machines[0][0] == 1
        assert inst.durations[0][0] == 44
        for j in range(10):
            assert sorted(inst.machines[j]) == list(range(5))

    def test_ta01_zero_based(self, raw_dir):
        inst = parse_jobshop_file(raw_dir / "ta01.txt")
        assert (inst.n, inst.m) == (10, 10)
        assert inst.machines[0][0] == 0
        assert inst.durations[0][0] == 43

    def test_comment_lines_skipped(self, tmp_path):
        path = tmp_path / "with_comment.txt"
        path.write_text("# comment\n10 5\n" + LA01_RAW.split("\n", 1)[1], encoding="utf-8")
        inst = parse_jobshop_file(path)
        assert inst.n == 10


class TestErrorHandling:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="空"):
            parse_jobshop_file(path)

    def test_bad_header(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("10 5 3\n2 44 3 5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="首行"):
            parse_jobshop_file(path)

    def test_row_length_mismatch(self, tmp_path):
        path = tmp_path / "badrow.txt"
        path.write_text("1 2\n0 1 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="行长度"):
            parse_jobshop_file(path)

    def test_machine_number_not_normalizable(self, tmp_path):
        path = tmp_path / "badnum.txt"
        path.write_text("1 2\n3 1 4 2\n", encoding="utf-8")  # 机器号 3/4，min=3，无法归一化
        with pytest.raises(ValueError, match="归一化"):
            parse_jobshop_file(path)
