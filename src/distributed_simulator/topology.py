from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from distributed_simulator.config import Topology


@dataclass(frozen=True)
class MixGroup:
    ranks: tuple[int, ...]
    weight: float


def communication_schedule(topology: Topology, workers: int) -> tuple[MixGroup, ...]:
    if workers < 1 or workers & (workers - 1):
        raise ValueError(f"workers must be a positive power of two, got {workers}")

    if topology == Topology.COMPLETE:
        return (MixGroup(tuple(range(workers)), 1.0 / workers),)
    if topology == Topology.EXP:
        return tuple(_exp_groups(workers))
    if topology == Topology.RING:
        return tuple(_ring_groups(workers))
    raise ValueError(f"unsupported topology: {topology}")


def groups_for_rank(schedule: Iterable[MixGroup], rank: int) -> tuple[MixGroup, ...]:
    return tuple(group for group in schedule if rank in group.ranks)


def _ring_groups(workers: int) -> Iterable[MixGroup]:
    if workers == 1:
        yield MixGroup((0,), 1.0)
        return

    for start in (0, 1):
        for rank in range(start, workers, 2):
            yield MixGroup(tuple(sorted((rank, (rank + 1) % workers))), 0.5)


def _exp_groups(workers: int) -> Iterable[MixGroup]:
    if workers == 1:
        yield MixGroup((0,), 1.0)
        return

    offset = 1
    while offset < workers:
        for rank in range(workers):
            peer = rank ^ offset
            if rank < peer:
                yield MixGroup((rank, peer), 0.5)
        offset <<= 1
