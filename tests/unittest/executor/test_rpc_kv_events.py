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
from collections.abc import Callable, Iterator
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import IterationResult
from tensorrt_llm.executor.rpc.rpc_client import RPCClient
from tensorrt_llm.executor.rpc.rpc_common import RPCRequest, get_unique_ipc_addr
from tensorrt_llm.executor.rpc.rpc_server import RPCServer
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

pytestmark = pytest.mark.cpu_only
_Proxy = GenerationExecutorProxy | GenerationExecutorRpcProxy


class _KvEventApp:
    """Destructively drain events before a controllable RPC reply."""

    def __init__(self) -> None:
        self.records = ['{"event_id": 1}', '{"event_id": 2}']
        self.timeouts: list[float | None] = []
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def publish(self, event_id: int) -> None:
        with self.lock:
            self.records.append(json.dumps({"event_id": event_id}))

    async def fetch_kv_cache_events_wait_async(self, timeout: float | None = None) -> list[str]:
        with self.lock:
            records, self.records = self.records, []
            self.timeouts.append(timeout)
        self.started.set()

        async def wait_for_release() -> None:
            while not self.release.is_set():
                await asyncio.sleep(0.005)

        try:
            # Also bound failures when running against a blocking implementation.
            await asyncio.wait_for(wait_for_release(), 3)
        except asyncio.TimeoutError:
            pass
        finally:
            self.finished.set()
        if self.fail:
            raise ValueError("KV event fetch failed")
        return records


@pytest.fixture
def rpc_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[RPCClient, _KvEventApp]]:
    app = _KvEventApp()
    with RPCServer(app) as server:
        server.bind(get_unique_ipc_addr())
        server.start()
        with RPCClient(server.address, hmac_key=server.hmac_key, timeout=5) as client:
            real_put = client._client_socket.put_async

            async def checked_put(request: RPCRequest) -> None:
                assert asyncio.get_running_loop() is client._loop
                await real_put(request)

            monkeypatch.setattr(client._client_socket, "put_async", checked_put)
            try:
                yield client, app
            finally:
                app.release.set()


@pytest.fixture(params=[GenerationExecutorProxy, GenerationExecutorRpcProxy])
def proxy(
    request: pytest.FixtureRequest,
    rpc_env: tuple[RPCClient, _KvEventApp],
    monkeypatch: pytest.MonkeyPatch,
) -> _Proxy:
    instance = object.__new__(request.param)
    instance.shutdown = lambda: None
    # The base initializes public-method state; no engine resources exist to finalize.
    with monkeypatch.context() as scoped:
        scoped.setattr("tensorrt_llm.executor.executor.atexit.register", lambda *args: None)
        GenerationExecutor.__init__(instance, is_llm_executor=True)
    instance.rpc_client = rpc_env[0]
    return instance


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 2)


async def _collect(result: IterationResult) -> list[dict]:
    return [event async for event in result]


async def _read(proxy: _Proxy, synchronous: bool) -> list[dict]:
    if synchronous:
        return await asyncio.to_thread(proxy.get_kv_events, 0)
    return await _collect(proxy.aget_kv_events(0))


@pytest.mark.parametrize("timeout", [0, None, 2])
@pytest.mark.asyncio
async def test_lazy_fetch_keeps_caller_responsive(rpc_env, proxy: _Proxy, timeout) -> None:
    client, app = rpc_env
    app.release.clear()
    result = proxy.aget_kv_events(timeout)
    assert isinstance(result, IterationResult)
    assert app.timeouts == [], "constructing the iterator must not fetch"
    pending = asyncio.create_task(_collect(result))
    await _wait_until(app.started.is_set)
    try:
        assert not app.finished.is_set()
        assert not pending.done()
        assert await asyncio.wait_for(asyncio.sleep(0, result="heartbeat"), 1) == "heartbeat"
    finally:
        app.release.set()
    assert await asyncio.wait_for(pending, 2) == [{"event_id": 1}, {"event_id": 2}]
    assert app.timeouts == [timeout]
    assert await _collect(result) == []
    assert not client._pending_futures


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.asyncio
async def test_cancelled_consumer_preserves_destructively_drained_batch(
    rpc_env, proxy: _Proxy, synchronous: bool
) -> None:
    _, app = rpc_env
    app.release.clear()
    pending = asyncio.create_task(proxy.aget_kv_events(0).__anext__())
    await _wait_until(app.started.is_set)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not app.finished.is_set()
    app.release.set()
    assert await asyncio.wait_for(_read(proxy, synchronous), 2) == [
        {"event_id": 1},
        {"event_id": 2},
    ]
    assert app.timeouts == [0], "resuming must adopt the original destructive fetch"
    assert await _read(proxy, synchronous) == []


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.asyncio
async def test_partial_batch_is_drained_before_another_fetch(
    rpc_env, proxy: _Proxy, synchronous: bool
) -> None:
    _, app = rpc_env
    result = proxy.aget_kv_events(0)
    assert await result.__anext__() == {"event_id": 1}
    app.publish(3)
    assert await _read(proxy, synchronous) == [{"event_id": 2}]
    assert app.timeouts == [0]
    assert await _read(proxy, synchronous) == [{"event_id": 3}]
    assert app.timeouts == [0, 0]


@pytest.mark.asyncio
async def test_distinct_iterators_share_pending_fetch_without_duplicate_events(
    rpc_env, proxy: _Proxy
) -> None:
    _, app = rpc_env
    app.release.clear()
    first, second = proxy.aget_kv_events(0), proxy.aget_kv_events(0)
    first_pending = asyncio.create_task(first.__anext__())
    await _wait_until(app.started.is_set)
    second_pending = asyncio.create_task(second.__anext__())
    await asyncio.sleep(0)
    try:
        assert app.timeouts == [0]
        assert not second_pending.done()
    finally:
        app.release.set()
    records = await asyncio.wait_for(asyncio.gather(first_pending, second_pending), 2)
    assert sorted(record["event_id"] for record in records) == [1, 2]
    assert await _collect(first) == []
    assert await _collect(second) == []
    assert app.timeouts == [0]


@pytest.mark.asyncio
async def test_same_iterator_rejects_overlapping_consumption(rpc_env, proxy: _Proxy) -> None:
    _, app = rpc_env
    app.release.clear()
    result = proxy.aget_kv_events(0)
    pending = asyncio.create_task(result.__anext__())
    await _wait_until(app.started.is_set)
    try:
        with pytest.raises(RuntimeError, match="[Cc]oncurrent"):
            await result.__anext__()
    finally:
        app.release.set()
    assert await pending == {"event_id": 1}
    assert await _collect(result) == [{"event_id": 2}]


@pytest.mark.asyncio
async def test_rpc_failure_does_not_discard_buffered_events(rpc_env, proxy: _Proxy) -> None:
    _, app = rpc_env
    assert await proxy.aget_kv_events(0).__anext__() == {"event_id": 1}
    app.fail = True
    assert await _collect(proxy.aget_kv_events(0)) == [{"event_id": 2}]
    assert app.timeouts == [0]
    assert await _collect(proxy.aget_kv_events(0)) == []
    app.fail = False
    app.publish(3)
    assert await _collect(proxy.aget_kv_events(0)) == [{"event_id": 3}]


@pytest.mark.asyncio
async def test_transport_timeout_finishes_and_allows_later_fetch(rpc_env, proxy: _Proxy) -> None:
    client, app = rpc_env
    app.records = []
    app.release.clear()
    client._timeout = 0.2
    pending = asyncio.create_task(_collect(proxy.aget_kv_events(0)))
    await _wait_until(app.started.is_set)
    assert await asyncio.wait_for(pending, 2) == []
    assert not client._pending_futures
    app.release.set()
    await _wait_until(app.finished.is_set)
    client._timeout = 5
    app.publish(3)
    assert await _collect(proxy.aget_kv_events(0)) == [{"event_id": 3}]


@pytest.mark.asyncio
async def test_malformed_event_is_not_silently_swallowed(rpc_env, proxy: _Proxy) -> None:
    _, app = rpc_env
    app.records = ["not JSON", '{"event_id": 2}']
    with pytest.raises(json.JSONDecodeError):
        await proxy.aget_kv_events(0).__anext__()
    assert await _collect(proxy.aget_kv_events(0)) == [{"event_id": 2}]
    assert app.timeouts == [0]


@pytest.mark.asyncio
async def test_sync_decode_failure_preserves_other_events(rpc_env, proxy: _Proxy) -> None:
    _, app = rpc_env
    app.records = ['{"event_id": 1}', "not JSON", '{"event_id": 2}']
    with pytest.raises(json.JSONDecodeError):
        await asyncio.to_thread(proxy.get_kv_events, 0)
    assert await _collect(proxy.aget_kv_events(0)) == [{"event_id": 1}, {"event_id": 2}]
    assert app.timeouts == [0]


@pytest.mark.asyncio
async def test_proxy_shutdown_finishes_pending_consumer(rpc_env, proxy: _Proxy) -> None:
    client, app = rpc_env
    # Only absent engine lifecycle resources are stubbed; shutdown and RPC close are real.
    if isinstance(proxy, GenerationExecutorRpcProxy):
        proxy._shutdown_event = threading.Event()
        proxy.shutdown_remote = lambda: None
        proxy.main_loop = proxy.main_loop_task_obj = proxy.main_loop_thread = None
        proxy.mpi_session = None
    else:
        proxy.workers_started = proxy.doing_shutdown = True
        proxy._engine_dead = proxy._owns_mpi_session = False
        proxy.mpi_futures = []
        proxy.dispatch_result_thread = None
        proxy.request_queue = Mock()
        proxy.worker_init_status_queue = Mock()
        proxy.result_queue = Mock()
        proxy._resource_governor_queue = proxy.profile_ack_queue = None
    app.release.clear()
    pending = asyncio.create_task(_collect(proxy.aget_kv_events(0)))
    await _wait_until(app.started.is_set)
    rpc_thread = client._loop_thread
    await asyncio.to_thread(type(proxy).shutdown, proxy)
    assert await asyncio.wait_for(pending, 2) == []
    assert not client._pending_futures
    assert not rpc_thread.is_alive()


@pytest.mark.asyncio
async def test_unavailable_client_returns_empty(proxy: _Proxy) -> None:
    proxy.rpc_client = None
    assert await _collect(proxy.aget_kv_events(0)) == []
    assert proxy.get_kv_events(0) == []


@pytest.mark.asyncio
async def test_empty_batch_has_no_second_local_timeout(rpc_env, proxy: _Proxy) -> None:
    _, app = rpc_env
    app.records = []
    assert await asyncio.wait_for(_collect(proxy.aget_kv_events(30)), 1) == []
    assert app.timeouts == [30]


@pytest.mark.asyncio
async def test_async_result_supports_synchronous_drain(proxy: _Proxy) -> None:
    result = proxy.aget_kv_events(0)
    assert isinstance(result, IterationResult)
    assert await asyncio.to_thread(result.get_results) == [{"event_id": 1}, {"event_id": 2}]
    assert result.get_results() == []
