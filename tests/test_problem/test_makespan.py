"""makespan 计算与最左调度测试。"""

from problem.instance import Instance
from problem.makespan import compute_start_times, makespan_from_machine_order, makespan_from_starts
from problem.validator import validate


def two_by_two_all_ones():
    """n=2, m=2 全 1 时间：最优 makespan = 2。"""
    return Instance(
        instance_id="manual-2x2",
        n=2,
        m=2,
        machines=[[0, 1], [1, 0]],
        durations=[[1, 1], [1, 1]],
    )


class TestMakespanFromStarts:
    def test_simple(self):
        inst = two_by_two_all_ones()
        starts = [[0, 1], [0, 1]]  # m0: (0,0)[0,1)、(1,1)[1,2)；m1: (0,1)[1,2)、(1,0)[0,1)
        assert makespan_from_starts(inst, starts) == 2

    def test_single_job(self):
        inst = Instance(instance_id="manual-1x3", n=1, m=3, machines=[[0, 1, 2]], durations=[[2, 3, 4]])
        starts = [[0, 2, 5]]
        assert makespan_from_starts(inst, starts) == 9  # 2+3+4，串行


class TestComputeStartTimes:
    def test_parallel_jobs(self):
        inst = two_by_two_all_ones()
        order = [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]
        starts = compute_start_times(inst, order)
        assert makespan_from_starts(inst, starts) == 2
        assert starts[0][0] == 0 and starts[1][0] == 0

    def test_sequential_on_same_machine(self):
        inst = Instance(instance_id="manual-3x1", n=3, m=1, machines=[[0], [0], [0]], durations=[[2], [3], [4]])
        order = [[(0, 0), (1, 0), (2, 0)]]
        starts = compute_start_times(inst, order)
        assert starts == [[0], [2], [5]]
        assert makespan_from_starts(inst, starts) == 9

    def test_job_precedence_enforced(self):
        # 排列把工件 0 的工序 1 列在机器 1 首位，最左调度仍须满足工件顺序
        inst = two_by_two_all_ones()
        order = [[(0, 0)], [(0, 1)]]
        starts = compute_start_times(inst, order)
        assert validate(inst, starts).valid
        assert starts[0][1] >= starts[0][0] + 1

    def test_machine_order_makespan(self):
        inst = two_by_two_all_ones()
        assert makespan_from_machine_order(inst, [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]) == 2

    def test_cycle_in_order_raises(self):
        """排列与工件顺序矛盾（偏序图有环）时抛 ValueError。"""
        import pytest

        inst = two_by_two_all_ones()
        # m0: (0,0) → (1,1)；m1: (1,1) 之后 (0,0)？(0,0) 只在一台机器上；
        # 构造环：m0 上 (0,0)→(1,1)，m1 上 (1,1)→(1,0)→(0,1) 与工件顺序无环，
        # 直接构造含环例子：工件 0 两道工序分别排在互为前置的位置不可行——
        # 简便做法：把 (1,1) 与 (1,0) 的机器顺序反转制造与工件顺序矛盾
        inst2 = Instance(
            instance_id="manual-2x2",
            n=2, m=2,
            machines=[[0, 1], [0, 1]],  # 工件 1 两道工序都在 m0
            durations=[[1, 1], [1, 1]],
        )
        # 工件顺序要求 (1,0)→(1,1)；机器排列要求 (1,1)→(1,0) → 环
        order = [[(0, 0), (1, 1), (1, 0)], [(0, 1)]]
        with pytest.raises(ValueError, match="环"):
            compute_start_times(inst2, order)
