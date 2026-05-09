"""Multiprocessing-friendly shared-memory ndarray (FluxRT-compatible API)."""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Union

import numpy as np


class SharedTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type[np.generic] = np.uint8,
        name: str | None = None,
        create: bool = False,
    ) -> None:
        """
        Args:
            shape: tuple-like shape of the array.
            dtype: numpy dtype or something convertible (e.g. np.float32).
            name: name of existing shared memory (required if create=False).
            create: if True, create a new SharedMemory block.
        """
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape, dtype=np.int64) * self.dtype.itemsize)

        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)
            self.name = self.shm.name
        else:
            if name is None:
                raise ValueError("name must be provided when create=False")
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name

        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def copy_from(self, tensor: Union["SharedTensor", np.ndarray]) -> None:
        if isinstance(tensor, SharedTensor):
            src = tensor.array
        else:
            src = np.asarray(tensor)

        if src.shape != self.shape:
            raise ValueError(f"Shape mismatch: source {src.shape} != target {self.shape}")

        if src.dtype != self.dtype:
            src = src.astype(self.dtype, copy=False)

        np.copyto(self.array, src)

    def to_numpy(self) -> np.ndarray:
        return self.array.copy()

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def close_and_unlink(self) -> None:
        self.close()
        self.unlink()
