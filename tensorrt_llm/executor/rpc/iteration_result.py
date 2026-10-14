# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import threading
from queue import Empty

from ...logger import logger
from ..result import IterationResult
from .rpc_client import RemoteCall
from .rpc_common import RPCCancelled, RPCError


class _RpcIterationResult(IterationResult):
    """Lazily fetch and drain one RPC batch of iteration statistics or events.

    The RPC client owns socket I/O on its private loop. Consumption runs on the
    caller's loop, and no RPC is started until the result is consumed.
    Each batch makes one fetch attempt, including on failure. Explicitly
    resetting the result appends a new batch after any unconsumed records.
    Records remain in the executor's queue when a caller stops consuming early.
    """

    def __init__(self, remote_call: RemoteCall, backlog: IterationResult) -> None:
        super().__init__()
        self._remote_call = remote_call
        self.queue = backlog.queue
        self.aqueue = backlog.aqueue
        self._fetched = False
        self._consumer_lock = threading.Lock()

    def set_timeout(self, timeout: float | None) -> None:
        self._timeout = timeout
        # This is the worker's polling timeout, not the transport deadline.
        self._remote_call.kwargs["timeout"] = timeout

    def mark_undone(self) -> None:
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError("Cannot reset an iteration result during consumption")
        try:
            self._done = False
            self._fetched = False
        finally:
            self._consumer_lock.release()

    def _check_client(self) -> None:
        client = self._remote_call.client
        loop = client._loop
        if client._closed or loop is None or loop.is_closed() or not loop.is_running():
            raise RPCCancelled("RPC client event loop is not running")

    def _begin_consumption(self) -> None:
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent consumption of an iteration result is not supported")

    @staticmethod
    def _decode(record: str | bytes | bytearray | dict) -> dict:
        return record if isinstance(record, dict) else json.loads(record)

    def _pop_record(self) -> str | bytes | bytearray | dict:
        if self.aqueue is None:
            return self.queue.get(timeout=0)
        # RPC completion controls waiting; this queue only retains records.
        # Both consumption APIs use its atomic synchronous pop, so no queue
        # notification or wait on the original caller's loop is needed.
        try:
            return self.aqueue.unsafe_get()
        except asyncio.QueueEmpty:
            raise Empty from None

    async def _fetch_async(self) -> list[str | bytes | bytearray | dict]:
        self._check_client()
        loop = self._remote_call.client._loop
        coroutine = self._remote_call.remote_async()
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except RuntimeError:
            coroutine.close()
            self._check_client()
            raise
        wrapped = asyncio.wrap_future(future)
        try:
            while True:
                # A stopped RPC loop cannot complete its pending futures. Check
                # its lifetime while waiting without blocking the caller loop.
                await asyncio.wait((wrapped,), timeout=0.1)
                if wrapped.done():
                    return wrapped.result()
                self._check_client()
        finally:
            if not future.done() and not loop.is_closed() and loop.is_running():
                future.cancel()

    async def __anext__(self) -> dict:
        self._begin_consumption()
        try:
            if self._done:
                raise StopAsyncIteration
            if not self._fetched:
                self._fetched = True
                try:
                    records = await self._fetch_async()
                except asyncio.CancelledError:
                    self._done = True
                    raise
                except RPCError as error:
                    logger.debug(f"Error fetching iteration results via RPC: {error}")
                    records = []
                for record in records:
                    self.queue.put_nowait(record)
            try:
                record = self._pop_record()
            except Empty:
                self._done = True
                raise StopAsyncIteration from None
            return self._decode(record)
        finally:
            self._consumer_lock.release()

    def get_results(self) -> list[dict]:
        """Synchronously consume the same batch exposed by async iteration."""
        self._begin_consumption()
        try:
            if self._done:
                return []
            if not self._fetched:
                self._fetched = True
                try:
                    self._check_client()
                    records = self._remote_call.remote()
                except RPCError as error:
                    logger.debug(f"Error fetching iteration results via RPC: {error}")
                    records = []
                for record in records:
                    self.queue.put_nowait(record)
            results = []
            while True:
                try:
                    record = self._pop_record()
                except Empty:
                    self._done = True
                    return results
                results.append(self._decode(record))
        finally:
            self._consumer_lock.release()
