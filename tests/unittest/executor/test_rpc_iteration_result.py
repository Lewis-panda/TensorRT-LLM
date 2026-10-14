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

import pytest

from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import IterationResult
from tensorrt_llm.executor.rpc.rpc_client import RPCClient
from tensorrt_llm.executor.rpc.rpc_common import (
    RPCError,
    RPCRequest,
    RPCResponse,
    RPCTimeout,
    get_unique_ipc_addr,
)
from tensorrt_llm.executor.rpc.rpc_server import RPCServer
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

pytestmark = pytest.mark.cpu_only


class _ReplySocket:
    """Replace transport only; use real RemoteCall and RPCClient request handling."""

    def __init__(self, client: RPCClient) -> None:
        self.client = client
        self.records = ['{"iter": 1}', {"iter": 2}]
        self.error = None
        self.delay = 0.05
        self.requests = []
        self.handles = []
        self.reply_release = threading.Event()
        self.reply_release.set()

    async def put_async(self, request: RPCRequest) -> None:
        assert asyncio.get_running_loop() is self.client._loop
        self.requests.append(request)
        response = RPCResponse(request.request_id, result=self.records, error=self.error)
        self.handles.append(self.client._loop.call_later(self.delay, self._reply, response))

    def _reply(self, response: RPCResponse) -> None:
        if not self.reply_release.is_set():
            self.handles.append(self.client._loop.call_later(0.001, self._reply, response))
            return
        self.client._handle_regular_response(response)


@pytest.fixture
def rpc_client() -> Iterator[tuple[RPCClient, _ReplySocket]]:
    # The real constructor connects ZMQ and starts a reader. Here a deterministic
    # reply socket exercises the real client planner/future/exception paths.
    client = object.__new__(RPCClient)
    client._loop = asyncio.new_event_loop()
    client._closed = False
    client._server_stopped = False
    client._timeout = 0.5
    client._pending_futures = {}
    socket = _ReplySocket(client)
    client._client_socket = socket
    started = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(client._loop)
        client._loop.call_soon(started.set)
        client._loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    assert started.wait(2)
    try:
        yield client, socket
    finally:
        # A test may deliberately stop the loop with an in-flight call. Resume
        # it just to cancel/drain tasks, never leave the test's thread alive.
        if not client._loop.is_running():
            thread.join(timeout=2)
            started.clear()
            thread = threading.Thread(target=run_loop)
            thread.start()
            assert started.wait(2)

        async def drain() -> None:
            for handle in socket.handles:
                handle.cancel()
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(drain(), client._loop).result(timeout=2)
        client._loop.call_soon_threadsafe(client._loop.stop)
        thread.join(timeout=2)
        assert not thread.is_alive()
        client._loop.close()


def _proxy(
    proxy_class: type[GenerationExecutorProxy] | type[GenerationExecutorRpcProxy],
    client: RPCClient | None,
) -> GenerationExecutorProxy | GenerationExecutorRpcProxy:
    proxy = object.__new__(proxy_class)
    proxy.rpc_client = client
    proxy._is_llm_executor = True
    proxy._iter_stats_result = None
    proxy._iter_kv_events_result = None
    proxy.shutdown = lambda: None
    return proxy


async def _collect(result: IterationResult) -> list[dict]:
    return [record async for record in result]


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.parametrize("proxy_class", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize(
    ("method", "rpc_method"),
    [
        ("aget_stats", "fetch_stats_wait_async"),
        ("aget_kv_events", "fetch_kv_cache_events_wait_async"),
    ],
)
@pytest.mark.parametrize("timeout", [None, 0, 2])
@pytest.mark.asyncio
async def test_lazy_nonblocking_batch_and_timeout_forwarding(
    rpc_client, proxy_class, method, rpc_method, timeout
) -> None:
    client, socket = rpc_client
    socket.reply_release.clear()
    result = getattr(_proxy(proxy_class, client), method)(timeout)
    assert isinstance(result, IterationResult)
    assert socket.requests == [], "constructing the iterator must not fetch or block"

    pending = asyncio.create_task(_collect(result))
    await _wait_until(lambda: bool(socket.requests))
    # The gated RPC remains in flight while another task on the caller's loop
    # gets scheduled. Socket.put_async also asserts the client loop owns I/O.
    try:
        assert not pending.done()
        assert await asyncio.wait_for(asyncio.sleep(0, result="heartbeat"), 0.5) == "heartbeat"
    finally:
        socket.reply_release.set()
    assert await asyncio.wait_for(pending, 1) == [{"iter": 1}, {"iter": 2}]
    request = socket.requests[0]
    assert request.method_name == rpc_method
    assert request.kwargs["timeout"] == timeout
    assert request.timeout == client._timeout, "worker polling and RPC deadlines are separate"
    assert len(socket.requests) == 1
    assert await _collect(result) == []


@pytest.mark.asyncio
async def test_empty_batch_does_not_wait_a_second_local_timeout(rpc_client) -> None:
    client, socket = rpc_client
    socket.records = []
    result = _proxy(GenerationExecutorProxy, client).aget_stats(30)
    assert await asyncio.wait_for(_collect(result), 1) == []


@pytest.mark.parametrize("error", [RPCError("remote failure"), RPCTimeout("deadline")])
@pytest.mark.asyncio
async def test_rpc_errors_keep_empty_result_behavior(rpc_client, error) -> None:
    client, socket = rpc_client
    socket.error = error
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    assert await _collect(result) == []
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_transport_timeout_cleans_pending_request(rpc_client) -> None:
    client, socket = rpc_client
    socket.delay = 60
    client._timeout = 0.01
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    assert await asyncio.wait_for(_collect(result), 1) == []
    assert not client._pending_futures


@pytest.mark.asyncio
async def test_decode_error_is_not_swallowed_as_rpc_error(rpc_client) -> None:
    client, socket = rpc_client
    socket.records = ["not JSON"]
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    with pytest.raises(json.JSONDecodeError):
        await result.__anext__()


@pytest.mark.asyncio
async def test_local_error_propagates_without_automatic_retry(rpc_client) -> None:
    client, socket = rpc_client
    socket.error = ValueError("local failure")
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    with pytest.raises(ValueError, match="local failure"):
        await result.__anext__()
    assert await _collect(result) == []
    assert len(socket.requests) == 1


@pytest.mark.parametrize("record", [b'{"iter": 1}', bytearray(b'{"iter": 1}')])
@pytest.mark.asyncio
async def test_json_bytes_keep_iteration_result_decoding(rpc_client, record) -> None:
    client, socket = rpc_client
    socket.records = [record]
    assert await _collect(_proxy(GenerationExecutorProxy, client).aget_stats(0)) == [{"iter": 1}]


@pytest.mark.asyncio
async def test_cancellation_propagates_and_releases_pending_future(rpc_client) -> None:
    client, socket = rpc_client
    socket.delay = 60
    client._timeout = None
    result = _proxy(GenerationExecutorProxy, client).aget_stats(None)
    pending = asyncio.create_task(result.__anext__())
    await _wait_until(lambda: bool(client._pending_futures))
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await _wait_until(lambda: not client._pending_futures)
    with pytest.raises(StopAsyncIteration):
        await result.__anext__()
    assert len(socket.requests) == 1


@pytest.mark.asyncio
async def test_same_iterator_concurrent_consumers_do_not_refetch(rpc_client) -> None:
    client, socket = rpc_client
    socket.reply_release.clear()
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    pending = asyncio.create_task(result.__anext__())
    await _wait_until(lambda: bool(client._pending_futures))
    try:
        with pytest.raises(RuntimeError, match="Concurrent consumption"):
            await result.__anext__()
        with pytest.raises(RuntimeError, match="Concurrent consumption"):
            result.get_results()
        with pytest.raises(RuntimeError, match="during consumption"):
            result.mark_undone()
    finally:
        socket.reply_release.set()
    assert await pending == {"iter": 1}
    assert await _collect(result) == [{"iter": 2}]
    assert len(socket.requests) == 1


@pytest.mark.parametrize("stop_loop", [False, True])
@pytest.mark.asyncio
async def test_client_shutdown_during_fetch_does_not_hang(rpc_client, stop_loop) -> None:
    client, socket = rpc_client
    socket.delay = 60
    client._timeout = None
    pending = asyncio.create_task(_collect(_proxy(GenerationExecutorProxy, client).aget_stats(0)))
    await _wait_until(lambda: bool(client._pending_futures))
    if stop_loop:
        client._loop.call_soon_threadsafe(client._loop.stop)
        await _wait_until(lambda: not client._loop.is_running())
    else:
        client._closed = True
    assert await asyncio.wait_for(pending, 1) == []
    if not stop_loop:
        await _wait_until(lambda: not client._pending_futures)


@pytest.mark.asyncio
async def test_closed_client_before_consumption_never_schedules(rpc_client) -> None:
    client, socket = rpc_client
    client._closed = True
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    assert await _collect(result) == []
    assert socket.requests == []


@pytest.mark.asyncio
async def test_closed_loop_before_consumption_never_schedules() -> None:
    client = object.__new__(RPCClient)
    client._closed = False
    client._loop = asyncio.new_event_loop()
    client._loop.close()
    assert await _collect(_proxy(GenerationExecutorProxy, client).aget_stats(0)) == []


def test_synchronous_consumption_and_explicit_reset(rpc_client) -> None:
    client, socket = rpc_client
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    assert isinstance(result, IterationResult)
    assert result.get_results() == [{"iter": 1}, {"iter": 2}]
    assert result.get_results() == []
    result.mark_undone()
    result.set_timeout(None)
    assert result.get_results() == [{"iter": 1}, {"iter": 2}]
    assert len(socket.requests) == 2
    assert socket.requests[-1].kwargs["timeout"] is None


@pytest.mark.asyncio
async def test_independent_calls_and_partial_sync_drain(rpc_client) -> None:
    client, socket = rpc_client
    proxy = _proxy(GenerationExecutorProxy, client)
    first = proxy.aget_stats(0)
    second = proxy.aget_stats(0)
    assert first is not second
    assert await first.__anext__() == {"iter": 1}
    assert first.get_results() == [{"iter": 2}]
    assert await _collect(second) == [{"iter": 1}, {"iter": 2}]
    assert len(socket.requests) == 2


@pytest.mark.parametrize("proxy_class", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("method", ["aget_stats", "aget_kv_events"])
@pytest.mark.parametrize("synchronous_drain", [False, True])
@pytest.mark.asyncio
async def test_next_call_retains_records_after_early_exit(
    rpc_client, proxy_class, method, synchronous_drain
) -> None:
    client, socket = rpc_client
    proxy = _proxy(proxy_class, client)
    socket.records = ['{"iter": 1}', '{"iter": 2}']
    async for record in getattr(proxy, method)(0.01):
        assert record == {"iter": 1}
        break

    socket.records = ['{"iter": 3}']
    result = getattr(proxy, method)(0.01)
    records = result.get_results() if synchronous_drain else await _collect(result)
    assert records == [{"iter": 2}, {"iter": 3}]
    assert len(socket.requests) == 2


@pytest.mark.asyncio
async def test_next_call_rpc_error_keeps_buffered_records(rpc_client) -> None:
    client, socket = rpc_client
    proxy = _proxy(GenerationExecutorProxy, client)
    assert await proxy.aget_stats(0).__anext__() == {"iter": 1}
    socket.error = RPCError("remote failure")
    assert await _collect(proxy.aget_stats(0)) == [{"iter": 2}]


@pytest.mark.asyncio
async def test_cancellation_keeps_shared_backlog_for_next_call(rpc_client) -> None:
    client, socket = rpc_client
    proxy = _proxy(GenerationExecutorProxy, client)
    assert await proxy.aget_kv_events(0).__anext__() == {"iter": 1}
    socket.delay = 60
    pending = asyncio.create_task(proxy.aget_kv_events(0).__anext__())
    await _wait_until(lambda: bool(client._pending_futures))
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await _wait_until(lambda: not client._pending_futures)

    socket.delay = 0.05
    socket.records = ['{"iter": 3}']
    assert await _collect(proxy.aget_kv_events(0)) == [{"iter": 2}, {"iter": 3}]


def test_shared_backlog_does_not_depend_on_original_caller_loop(rpc_client) -> None:
    client, socket = rpc_client

    async def consume_first() -> GenerationExecutorProxy:
        proxy = _proxy(GenerationExecutorProxy, client)
        assert isinstance(proxy, GenerationExecutorProxy)
        assert await proxy.aget_stats(0).__anext__() == {"iter": 1}
        return proxy

    proxy = asyncio.run(consume_first())
    socket.records = ['{"iter": 3}']
    assert proxy.aget_stats(0).get_results() == [{"iter": 2}, {"iter": 3}]


@pytest.mark.asyncio
async def test_explicit_reset_preserves_unconsumed_records(rpc_client) -> None:
    client, socket = rpc_client
    result = _proxy(GenerationExecutorProxy, client).aget_stats(0)
    assert await result.__anext__() == {"iter": 1}
    result.mark_undone()
    socket.records = ['{"iter": 3}']
    assert await _collect(result) == [{"iter": 2}, {"iter": 3}]
    assert len(socket.requests) == 2


@pytest.mark.parametrize("proxy_class", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("method", ["aget_stats", "aget_kv_events"])
@pytest.mark.asyncio
async def test_unavailable_client_returns_empty(proxy_class, method) -> None:
    assert await _collect(getattr(_proxy(proxy_class, None), method)(0)) == []


class _StatsApp:
    """Expose bounded, controllable RPC endpoints without constructing an engine."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.worker_timeout: float | None = None

    async def _fetch(self, timeout: float | None) -> list[str | dict]:
        self.worker_timeout = timeout
        self.started.set()

        async def wait_for_release() -> None:
            while not self.release.is_set():
                await asyncio.sleep(0.005)

        try:
            await asyncio.wait_for(wait_for_release(), timeout=2)
        except asyncio.TimeoutError:
            # Bound the old synchronous implementation's failure as well.
            pass
        self.finished.set()
        return ['{"iter": 1}', {"iter": 2}]

    async def fetch_stats_wait_async(self, timeout: float | None = None) -> list[str | dict]:
        return await self._fetch(timeout)

    async def fetch_kv_cache_events_wait_async(
        self, timeout: float | None = None
    ) -> list[str | dict]:
        return await self._fetch(timeout)


@pytest.fixture
def real_rpc_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[RPCClient, _StatsApp]]:
    app = _StatsApp()
    # The same bind/start/context lifecycle as test_rpc.RpcServerWrapper, using
    # the actual HMAC ZMQ transport, client reader and server method dispatch.
    with RPCServer(app) as server:
        server.bind(get_unique_ipc_addr())
        server.start()
        with RPCClient(server.address, hmac_key=server.hmac_key, timeout=3) as client:
            real_put = client._client_socket.put_async

            async def checked_put(request: RPCRequest) -> None:
                assert asyncio.get_running_loop() is client._loop
                await real_put(request)

            monkeypatch.setattr(client._client_socket, "put_async", checked_put)
            try:
                yield client, app
            finally:
                app.release.set()


@pytest.mark.parametrize("proxy_class", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("method", ["aget_stats", "aget_kv_events"])
@pytest.mark.asyncio
async def test_real_transport_keeps_caller_loop_responsive(
    real_rpc_client, proxy_class, method
) -> None:
    client, app = real_rpc_client
    result = getattr(_proxy(proxy_class, client), method)(0.25)
    assert isinstance(result, IterationResult)
    assert not app.started.is_set(), "constructing the iterator must not submit an RPC"
    pending = asyncio.create_task(_collect(result))
    await _wait_until(app.started.is_set)
    try:
        assert not pending.done()
        assert await asyncio.wait_for(asyncio.sleep(0, result="heartbeat"), 0.5) == "heartbeat"
    finally:
        app.release.set()
    assert await asyncio.wait_for(pending, 2) == [{"iter": 1}, {"iter": 2}]
    assert app.worker_timeout == 0.25
    assert not client._pending_futures


@pytest.mark.parametrize("proxy_class", [GenerationExecutorProxy, GenerationExecutorRpcProxy])
@pytest.mark.parametrize("method", ["aget_stats", "aget_kv_events"])
@pytest.mark.asyncio
async def test_real_transport_cancellation_drains_client_pending_requests(
    real_rpc_client, proxy_class, method
) -> None:
    client, app = real_rpc_client
    result = getattr(_proxy(proxy_class, client), method)(0)
    pending = asyncio.create_task(result.__anext__())
    await _wait_until(lambda: app.started.is_set() and bool(client._pending_futures))
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await _wait_until(lambda: not client._pending_futures)
    with pytest.raises(StopAsyncIteration):
        await result.__anext__()
    # Cancellation releases the client wait. The remote method is allowed to
    # finish; this is not a server-side cancellation protocol.
    app.release.set()
    await _wait_until(app.finished.is_set)
