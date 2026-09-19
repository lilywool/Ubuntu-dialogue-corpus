"""Local parallel execution backend for pipeline stages."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def recommended_workers(row_count: int, *, max_workers: int | None = 4) -> int:
    """Choose a conservative process count from local CPU capacity and work size.

    The reviewed lexicons are large module-level objects. Capping the default
    prevents a high-core-count Windows machine from duplicating them into
    dozens of worker processes.
    """
    cpu_count = os.cpu_count() or 1
    available = max(1, cpu_count - 1)
    if max_workers is not None:
        available = min(available, max_workers)
    if row_count < 1_000:
        return 1
    return min(available, row_count)


def map_rows(
    func: Callable[[T], R],
    values: Iterable[T],
    workers: int = 1,
    *,
    chunksize: int = 256,
) -> list[R]:
    """Map a top-level worker without duplicating the full input in memory."""
    if workers <= 1:
        return [func(value) for value in values]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, values, chunksize=max(1, chunksize)))


def iter_batches(values: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Yield bounded lists from an iterable without copying the full input."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    iterator = iter(values)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def map_batches(
    func: Callable[[list[T]], list[R]],
    values: Iterable[T],
    workers: int = 1,
    *,
    batch_size: int = 128,
) -> list[R]:
    """Process bounded batches through the shared local execution backend."""
    return [
        result
        for batch_results in iter_batch_results(
            func,
            values,
            workers=workers,
            batch_size=batch_size,
        )
        for result in batch_results
    ]


def iter_batch_results(
    func: Callable[[list[T]], list[R]],
    values: Iterable[T],
    workers: int = 1,
    *,
    batch_size: int = 128,
) -> Iterable[list[R]]:
    """Yield processed batches so callers can avoid a full record-list copy."""
    batches = iter_batches(values, batch_size)
    if workers <= 1:
        for batch in batches:
            yield func(batch)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(func, batches, chunksize=1)
