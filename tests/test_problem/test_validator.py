"""可行性验证器测试：三条硬约束的违规检测。"""

from problem.instance import Instance
from problem.makespan import compute_start_times
from problem.validator import validate, validate_machine_order


def instance_2x2():
    return Instance(
        instance_id="manual-2x2",
        n=2,
        m=2,
        machines=[[0, 1], [1, 0]],
        durations=[[2, 3], [4, 1]],
    )


def valid_starts():
    inst = instance_2x2()
    order = [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]
    return compute_start_times(inst, order)


class TestValidSchedule:
    def test_left_shifted_schedule_is_valid(self):
        inst = instance_2x2()
        assert validate(inst, valid_starts()).valid

    def test_makespan_reported_when_valid(self):
        inst = instance_2x2()
        result = validate(inst, valid_starts())
        assert result.makespan is not None


class TestInvalidSchedules:
    def test_shape_wrong(self):
        inst = instance_2x2()
        result = validate(inst, [[0, 0]])  # 只有 1 行
        assert not result.valid
        assert "行数" in result.errors[0]

    def test_unassigned_operation(self):
        inst = instance_2x2()
        starts = valid_starts()
        starts[1][1] = -1
        result = validate(inst, starts)
        assert not result.valid
        assert any("未调度" in e for e in result.errors)

    def test_job_precedence_violation(self):
        inst = instance_2x2()
        starts = valid_starts()
        # 把工件 0 的工序 1 提前到工序 0 完成之前（同时开始即违规：工序 0 需 2 单位时间）
        starts[0][1] = starts[0][0]
        result = validate(inst, starts)
        assert not result.valid
        assert any("早于" in e and "工序 1" in e for e in result.errors)

    def test_machine_overlap(self):
        inst = instance_2x2()
        # m0 上同时安排 (0,0) 与 (1,1)，时间重叠
        starts = [[0, 0], [0, 1]]
        result = validate(inst, starts)
        assert not result.valid
        assert any("冲突" in e for e in result.errors)

    def test_no_makespan_when_invalid(self):
        inst = instance_2x2()
        result = validate(inst, [[-1, 0], [0, 0]])
        assert result.makespan is None

    def test_bool_dunder(self):
        inst = instance_2x2()
        assert bool(validate(inst, valid_starts())) is True
        assert bool(validate(inst, [[-1, -1], [0, 0]])) is False


class TestMachineOrderValidation:
    def test_valid_order(self):
        inst = instance_2x2()
        order = [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]
        assert validate_machine_order(inst, order).valid

    def test_duplicate_operation(self):
        inst = instance_2x2()
        order = [[(0, 0), (0, 0)], [(1, 0), (0, 1)]]
        result = validate_machine_order(inst, order)
        assert not result.valid
        assert any("重复" in e for e in result.errors)

    def test_wrong_machine(self):
        inst = instance_2x2()
        order = [[(0, 1)], [(1, 0), (0, 0)]]  # (0,1) 在 m1，却列在 m0
        result = validate_machine_order(inst, order)
        assert not result.valid
        assert any("不在机器" in e for e in result.errors)

    def test_missing_operation(self):
        inst = instance_2x2()
        order = [[(0, 0)], [(1, 0), (0, 1)]]  # 漏掉 (1,1)
        result = validate_machine_order(inst, order)
        assert not result.valid
        assert any("漏掉" in e for e in result.errors)

    def test_length_mismatch(self):
        inst = instance_2x2()
        assert not validate_machine_order(inst, [[(0, 0)]]).valid
