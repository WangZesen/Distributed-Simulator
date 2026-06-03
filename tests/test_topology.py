from distributed_simulator.config import Topology
from distributed_simulator.topology import communication_schedule


def test_ring_schedule_for_power_of_two_workers() -> None:
    schedule = communication_schedule(Topology.RING, 4)
    assert [group.ranks for group in schedule] == [(0, 1), (2, 3), (1, 2), (0, 3)]
    assert all(group.weight == 0.5 for group in schedule)


def test_complete_schedule_averages_all_workers() -> None:
    schedule = communication_schedule(Topology.COMPLETE, 4)
    assert len(schedule) == 1
    assert schedule[0].ranks == (0, 1, 2, 3)
    assert schedule[0].weight == 0.25
