"""train/val/test 划分测试。"""

import pytest

from data.splits import load_splits, make_splits, save_splits


IDS = [f"gen-{i:05d}" for i in range(100)]


class TestMakeSplits:
    def test_default_ratios(self):
        splits = make_splits(IDS, seed=0)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10
        # 全部实例恰好划分一次
        assert sorted(splits["train"] + splits["val"] + splits["test"]) == sorted(IDS)

    def test_reproducible(self):
        a = make_splits(IDS, seed=42)
        b = make_splits(IDS, seed=42)
        assert a == b

    def test_seed_changes_split(self):
        a = make_splits(IDS, seed=1)
        b = make_splits(IDS, seed=2)
        assert a["train"] != b["train"]

    def test_empty_input(self):
        splits = make_splits([], seed=0)
        assert all(v == [] for v in splits.values())

    def test_bad_ratios(self):
        with pytest.raises(ValueError):
            make_splits(IDS, ratios=(0.5, 0.5, 0.5), seed=0)
        with pytest.raises(ValueError):
            make_splits(IDS, ratios=(1.2, -0.1, -0.1), seed=0)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        splits = make_splits(IDS, seed=0)
        save_splits(splits, tmp_path)
        assert (tmp_path / "train.json").exists()
        assert (tmp_path / "val.json").exists()
        assert (tmp_path / "test.json").exists()
        loaded = load_splits(tmp_path)
        assert loaded == splits

    def test_load_missing_returns_partial(self, tmp_path):
        (tmp_path / "train.json").write_text('["a"]', encoding="utf-8")
        loaded = load_splits(tmp_path)
        assert loaded["train"] == ["a"]
        assert "val" not in loaded
