"""阶段 2 测试：TAI 模板构建、排程 JSON 编解码、容错解析。"""

import json

import pytest

from model.format import build_tai, encode_schedule, parse_schedule
from problem.generator import generate_batch
from problem.validator import validate


@pytest.fixture(scope="module")
def instance():
    return generate_batch(3, 3, count=1, seed=0)[0]


@pytest.fixture(scope="module")
def good_starts(instance):
    from solver.heuristics import gt_schedule

    return gt_schedule(instance, rule="spt")


class TestBuildTai:
    def test_contains_instance_info(self, instance):
        text = build_tai(instance)
        assert "3 jobs x 3 machines" in text
        assert "Job 0" in text and "Job 2" in text
        assert instance.instance_id in text
        # 至少一道工序出现在模板中
        assert f"(m{instance.machines[0][0]},{instance.durations[0][0]})" in text

    def test_contains_rules(self, instance):
        text = build_tai(instance)
        assert "minimizes the makespan" in text
        assert "JSON array" in text

    def test_heuristic_hint_appended(self, instance):
        text = build_tai(instance, heuristic_hint="best-known makespan: 123")
        assert "Hint: best-known makespan: 123" in text

    def test_no_hint_by_default(self, instance):
        assert "Hint:" not in build_tai(instance)


class TestEncodeParseRoundtrip:
    def test_roundtrip(self, instance, good_starts):
        text = encode_schedule(instance, good_starts)
        result = parse_schedule(instance, text)
        assert result.ok, result.error
        assert result.starts == good_starts

    def test_encoded_json_well_formed(self, instance, good_starts):
        text = encode_schedule(instance, good_starts)
        assert text.startswith("[")
        assert text.endswith("]")
        assert '"job"' in text and '"op"' in text and '"start"' in text


class TestParseTolerance:
    def test_code_block_wrapped(self, instance, good_starts):
        json_text = encode_schedule(instance, good_starts)
        wrapped = f"Here is the schedule:\n```json\n{json_text}\n```\nHope this helps."
        result = parse_schedule(instance, wrapped)
        assert result.ok, result.error
        assert result.starts == good_starts

    def test_leading_trailing_text(self, instance, good_starts):
        json_text = encode_schedule(instance, good_starts)
        result = parse_schedule(instance, f"Sure! {json_text} Done.")
        assert result.ok

    def test_field_order_any(self, instance, good_starts):
        # 字段顺序打乱后仍可解析
        json_text = encode_schedule(instance, good_starts)
        records = json.loads(json_text)
        shuffled = json.dumps(
            [{"duration": r["duration"], "start": r["start"], "op": r["op"], "job": r["job"], "machine": r["machine"]} for r in records]
        )
        result = parse_schedule(instance, shuffled)
        assert result.ok, result.error


class TestParseErrors:
    def test_no_json(self, instance):
        result = parse_schedule(instance, "I cannot solve this.")
        assert not result.ok
        assert "未找到" in result.error

    def test_malformed_json(self, instance):
        result = parse_schedule(instance, "[{'job': 0,]")
        assert not result.ok
        assert "JSON 解析失败" in result.error

    def test_not_a_list(self, instance):
        result = parse_schedule(instance, '{"job": 0}')
        assert not result.ok
        assert "数组" in result.error

    def test_missing_fields(self, instance):
        result = parse_schedule(instance, '[{"job": 0, "op": 0}]')
        assert not result.ok
        assert "字段非法" in result.error

    def test_out_of_bounds_operation(self, instance):
        text = '[{"job": 99, "op": 0, "machine": 0, "start": 0, "duration": 1}]'
        result = parse_schedule(instance, text)
        assert not result.ok
        assert "越界" in result.error

    def test_duplicate_operation(self, instance):
        j0 = instance.machines[0][0]
        text = (
            f'[{{"job": 0, "op": 0, "machine": {j0}, "start": 0, "duration": {instance.durations[0][0]}}},'
            f' {{"job": 0, "op": 0, "machine": {j0}, "start": 5, "duration": {instance.durations[0][0]}}}]'
        )
        result = parse_schedule(instance, text)
        assert not result.ok
        assert "重复" in result.error

    def test_machine_mismatch(self, instance):
        wrong_machine = (instance.machines[0][0] + 1) % instance.m
        text = (
            f'[{{"job": 0, "op": 0, "machine": {wrong_machine}, "start": 0, "duration": {instance.durations[0][0]}}}]'
        )
        result = parse_schedule(instance, text)
        assert not result.ok
        assert "机器" in result.error

    def test_duration_mismatch(self, instance):
        j0 = instance.machines[0][0]
        text = f'[{{"job": 0, "op": 0, "machine": {j0}, "start": 0, "duration": 999}}]'
        result = parse_schedule(instance, text)
        assert not result.ok
        assert "加工时间" in result.error

    def test_missing_operations(self, instance):
        result = parse_schedule(instance, "[]")
        assert not result.ok
        assert "缺少工序" in result.error

    def test_negative_start(self, instance):
        j0 = instance.machines[0][0]
        text = f'[{{"job": 0, "op": 0, "machine": {j0}, "start": -3, "duration": {instance.durations[0][0]}}}]'
        result = parse_schedule(instance, text)
        assert not result.ok
        assert "开始时间" in result.error


class TestParseThenValidate:
    def test_parsed_schedule_passes_validator(self, instance, good_starts):
        text = encode_schedule(instance, good_starts)
        result = parse_schedule(instance, text)
        assert validate(instance, result.starts).valid
