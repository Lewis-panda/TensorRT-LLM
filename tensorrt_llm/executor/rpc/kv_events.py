# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import json
import threading
from collections import deque
from dataclasses import dataclass

from ...logger import logger
from ..result import IterationResult
from .rpc_client import RPCClient
from .rpc_common import RPCCancelled


@dataclass
class _PendingBatch:
    future: concurrent.futures.Future
    client: RPCClient

    def cancel_if_closed(self) -> None:
        # A completed reply remains usable even after the client has closed.
        if not self.future.done() and (self.client._closed or not self.client._loop.is_running()):
            self.future.cancel()


class _RpcKvEvents:
    """Own a destructive KV poll independently of the lifetime of its readers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: deque[str | dict] = deque()
        self._pending: _PendingBatch | None = None
        self._closed = False

    def result(self, client: RPCClient | None, timeout: float | None) -> IterationResult:
        return _KvEventsResult(self, client, timeout)

    def _start(self, client: RPCClient | None, timeout: float | None) -> _PendingBatch | None:
        with self._lock:
            if self._records:
                return None
            if self._pending is not None:
                return self._pending
            if self._closed or client is None or client._closed or not client._loop.is_running():
                raise RPCCancelled("KV events RPC client is unavailable")
            coroutine = client.fetch_kv_cache_events_wait_async(timeout=timeout).remote_async()
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, client._loop)
            except RuntimeError:
                coroutine.close()
                raise RPCCancelled("KV events RPC loop is closed") from None
            self._pending = _PendingBatch(future, client)
            return self._pending

    def _publish(self, pending: _PendingBatch) -> None:
        with self._lock:
            if self._pending is not pending:
                return
            try:
                self._records.extend(pending.future.result())
            except concurrent.futures.CancelledError:
                raise RPCCancelled("KV events RPC was cancelled") from None
            finally:
                self._pending = None

    def fetch(self, client: RPCClient | None, timeout: float | None) -> None:
        pending = self._start(client, timeout)
        if pending is not None:
            while not pending.future.done():
                pending.cancel_if_closed()
                concurrent.futures.wait([pending.future], timeout=0.1)
            self._publish(pending)

    async def fetch_async(self, client: RPCClient | None, timeout: float | None) -> None:
        pending = self._start(client, timeout)
        if pending is not None:
            wrapped = asyncio.wrap_future(pending.future)
            # A cancelled reader leaves this future alive. Retrieve a later
            # exception even if no reader remains on this caller's loop.
            wrapped.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )
            while not pending.future.done():
                pending.cancel_if_closed()
                # Unlike awaiting the future directly, cancelling this wait
                # does not cancel the shared destructive fetch.
                await asyncio.wait([wrapped], timeout=0.1)
            self._publish(pending)

    def pop(self) -> dict:
        with self._lock:
            record = self._records.popleft()
        return json.loads(record) if isinstance(record, str) else record

    def drain(self) -> list[dict]:
        with self._lock:
            records = []
            for index, record in enumerate(self._records):
                try:
                    records.append(json.loads(record) if isinstance(record, str) else record)
                except json.JSONDecodeError:
                    # None of the valid records have been delivered yet.
                    del self._records[index]
                    raise
            self._records.clear()
            return records

    def close(self) -> None:
        """Cancel outstanding I/O before the executor stops its RPC loop."""
        with self._lock:
            self._closed = True
            if self._pending is not None:
                self._pending.future.cancel()


class _KvEventsResult(IterationResult):
    def __init__(
        self, owner: _RpcKvEvents, client: RPCClient | None, timeout: float | None
    ) -> None:
        super().__init__()
        self._owner = owner
        self._client = client
        self._timeout = timeout
        self._fetched = False
        self._consumer = threading.Lock()

    def mark_undone(self) -> None:
        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("Concurrent consumption of a KV events iterator is not supported")
        try:
            self._done = False
            self._fetched = False
        finally:
            self._consumer.release()

    def get_results(self) -> list[dict]:
        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("Concurrent consumption of a KV events iterator is not supported")
        try:
            if self._done:
                return []
            if not self._fetched:
                self._fetched = True
                try:
                    self._owner.fetch(self._client, self._timeout)
                except Exception as error:
                    logger.debug(f"Error fetching KV events via RPC: {error}")
            records = self._owner.drain()
            self._done = True
            return records
        finally:
            self._consumer.release()

    async def __anext__(self) -> dict:
        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("Concurrent consumption of a KV events iterator is not supported")
        try:
            if self._done:
                raise StopAsyncIteration
            if not self._fetched:
                self._fetched = True
                try:
                    await self._owner.fetch_async(self._client, self._timeout)
                except Exception as error:
                    logger.debug(f"Error fetching KV events via RPC: {error}")
            try:
                return self._owner.pop()
            except IndexError:
                self._done = True
                raise StopAsyncIteration from None
        except asyncio.CancelledError:
            self._done = True
            raise
        finally:
            self._consumer.release()
