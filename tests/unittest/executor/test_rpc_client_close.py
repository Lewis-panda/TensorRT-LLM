# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
import time

import pytest

from tensorrt_llm.executor.rpc import RPCCancelled, RPCClient, RPCServer
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr

pytestmark = pytest.mark.cpu_only


class _App:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    async def wait(self) -> str:
        self.calls += 1
        self.started.set()
        deadline = time.monotonic() + 10
        while not self.release.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("test gate expired")
            await asyncio.sleep(0.005)
        return "completed"


@pytest.fixture
def rpc_pair():
    app = _App()
    server = RPCServer(app, async_run_task=True)
    address = get_unique_ipc_addr()
    server.bind(address)
    server.start()
    client = RPCClient(address, hmac_key=server.hmac_key, num_workers=1)
    threads = []

    def start_call(call):
        outcome = []

        def run():
            try:
                outcome.append(call())
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)
        return thread, outcome

    yield app, client, start_call
    app.release.set()
    closer, _ = start_call(client.close)
    closer.join(0.2)

    # A failing baseline must not strand executor threads at interpreter exit.
    async def rescue() -> None:
        await asyncio.sleep(0)
        for future, target_loop in list(client._pending_futures.values()):
            if target_loop is asyncio.get_running_loop() and not future.done():
                future.set_exception(RPCCancelled("test cleanup"))
        await asyncio.sleep(0.05)

    deadline = time.monotonic() + 2
    while client._pending_futures or any(thread.is_alive() for thread in threads):
        assert time.monotonic() < deadline, "test cleanup could not release RPC waiters"
        if client._loop.is_running():
            asyncio.run_coroutine_threadsafe(rescue(), client._loop).result(2)
        else:
            client._loop.run_until_complete(rescue())
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    server.shutdown()
    client._loop.close()


def _assert_closed(client: RPCClient) -> None:
    assert not client._pending_futures
    assert not client._loop_thread.is_alive()
    assert not any(thread.is_alive() for thread in client._executor._threads)


def test_close_finishes_active_and_queued_futures(rpc_pair):
    app, client, start_call = rpc_pair
    active = client.wait().remote_future()
    assert app.started.wait(2)
    queued = client.wait().remote_future()
    assert not queued.running()
    closer, outcome = start_call(client.close)
    closer.join(2)
    assert not closer.is_alive(), "close hung with an in-flight RPC"
    assert outcome == [None]
    for future in (active, queued):
        with pytest.raises(RPCCancelled):
            future.result(timeout=1)
    assert app.calls == 1
    _assert_closed(client)


def test_close_finishes_sync_waiter(rpc_pair):
    app, client, start_call = rpc_pair
    waiter, outcome = start_call(lambda: client.wait().remote())
    assert app.started.wait(2)
    closer, _ = start_call(client.close)
    closer.join(2)
    waiter.join(2)
    assert not closer.is_alive()
    assert not waiter.is_alive(), "sync waiter survived client.close()"
    assert len(outcome) == 1 and isinstance(outcome[0], RPCCancelled)
    _assert_closed(client)


@pytest.mark.asyncio
async def test_close_finishes_caller_loop_waiter(rpc_pair):
    app, client, _ = rpc_pair
    waiter = asyncio.create_task(client.wait().remote_async())
    assert await asyncio.to_thread(app.started.wait, 2)
    client.close()
    with pytest.raises(RPCCancelled):
        await asyncio.wait_for(waiter, 1)
    _assert_closed(client)


@pytest.mark.asyncio
async def test_close_cancels_send_before_pending_registration(rpc_pair, monkeypatch):
    app, client, _ = rpc_pair
    send_started = asyncio.Event()
    original_send = client._client_socket.put_async

    async def blocked_send(request):
        send_started.set()
        await asyncio.sleep(10)
        return await original_send(request)

    monkeypatch.setattr(client._client_socket, "put_async", blocked_send)
    waiter = asyncio.create_task(client.wait().remote_async())
    await asyncio.wait_for(send_started.wait(), 1)
    assert not client._pending_futures
    client.close()
    with pytest.raises(RPCCancelled):
        await asyncio.wait_for(waiter, 1)
    assert app.calls == 0
    _assert_closed(client)


@pytest.mark.asyncio
async def test_deferred_close_does_not_cancel_completed_caller(rpc_pair, monkeypatch):
    app, client, _ = rpc_pair
    loop = asyncio.get_running_loop()
    original_schedule = loop.call_soon_threadsafe
    delayed_results = []
    delayed_cancels = []
    response_ready = threading.Event()
    application_resumed = asyncio.Event()
    application_release = asyncio.Event()

    def defer(callback, *args, **kwargs):
        if callback.__name__ == "safe_set_result":
            delayed_results.append((callback, args))
            response_ready.set()
        elif callback.__name__ == "cancel_call":
            delayed_cancels.append((callback, args))
        else:
            return original_schedule(callback, *args, **kwargs)

    async def caller():
        result = await client.wait().remote_async()
        application_resumed.set()
        await application_release.wait()
        return result

    waiter = asyncio.create_task(caller())
    assert await asyncio.to_thread(app.started.wait, 2)
    monkeypatch.setattr(loop, "call_soon_threadsafe", defer)
    app.release.set()
    assert await asyncio.to_thread(response_ready.wait, 2)
    client.close()
    try:
        for callback, args in delayed_results:
            callback(*args)
        await asyncio.wait_for(application_resumed.wait(), 1)
        for callback, args in delayed_cancels:
            callback(*args)
        await asyncio.sleep(0)
        assert not waiter.done(), "close cancelled unrelated application work"
        application_release.set()
        assert await asyncio.wait_for(waiter, 1) == "completed"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    _assert_closed(client)


@pytest.mark.parametrize("mode", ["sync", "future", "async"])
def test_closed_client_rejects_new_calls(rpc_pair, mode):
    app, client, start_call = rpc_pair
    client.close()

    def invoke():
        if mode == "async":
            return asyncio.run(client.wait().remote_async())
        if mode == "future":
            return client.wait().remote_future()
        return client.wait().remote()

    caller, outcome = start_call(invoke)
    caller.join(1)
    assert not caller.is_alive(), "closed client accepted a blocking call"
    assert len(outcome) == 1 and isinstance(outcome[0], RPCCancelled)
    assert app.calls == 0
    _assert_closed(client)


def test_close_preserves_completed_future(rpc_pair):
    app, client, _ = rpc_pair
    app.release.set()
    result = client.wait().remote_future()
    assert result.result(timeout=2) == "completed"
    client.close()
    client.close()
    assert result.result(timeout=0) == "completed"
    _assert_closed(client)
