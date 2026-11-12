"""Minimal web UI: chat on the left, live trace on the right, demo controls on top.

Run with `make run` (uvicorn --factory errand.web.app:create_app) and open http://localhost:8000.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from errand.config import Paths
from errand.web.runtime import Runtime

STATIC = Path(__file__).parent / "static"
WATCH_INTERVAL_S = float(os.getenv("WATCH_INTERVAL_S", "3"))


class Message(BaseModel):
    text: str


class Reset(BaseModel):
    preset: str = "normal"
    seed: bool = True


class Key(BaseModel):
    key: str


class Broadcaster:
    """Fan-out from worker threads (agent, watcher) to every open browser stream."""

    def __init__(self):
        self.queues: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, item: dict) -> None:
        if self.loop is None:
            return
        for q in list(self.queues):
            self.loop.call_soon_threadsafe(q.put_nowait, item)


def create_app(runtime_factory=None, watch: bool = True) -> FastAPI:
    load_dotenv()
    hub = Broadcaster()
    rt: Runtime = (runtime_factory or (lambda publish: Runtime(Paths.traces.parent / "data",
                                                               Paths.traces / "web", publish)))(hub.publish)

    async def watch_loop():
        while True:
            await asyncio.sleep(WATCH_INTERVAL_S)
            try:
                await run_in_threadpool(rt.tick)
            except Exception as err:  # keep watching; the error is visible in the trace
                rt.tracer.emit("error", where="watcher_loop", error=repr(err))

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        task = asyncio.create_task(watch_loop()) if watch else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Errand Agent", lifespan=lifespan)
    app.state.runtime = rt
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return {**rt.state(), "history": rt.history(), "trace": rt.backlog(), "memories": rt.memories()}

    @app.get("/api/status")
    def status():
        return rt.state()

    @app.post("/api/message")
    def message(m: Message):
        if not m.text.strip():
            raise HTTPException(400, "empty message")
        return {**rt.say(m.text.strip()), "state": rt.state()}

    @app.get("/api/memories")
    def memories():
        return rt.memories()

    @app.post("/api/memories/forget")
    def forget(k: Key):
        return {"forgotten": rt.forget(k.key), "memories": rt.memories()}

    @app.get("/api/bookings")
    def bookings():
        return rt.bookings()

    @app.post("/api/demo/reset")
    def reset(r: Reset):
        try:
            rt.reset(r.preset, r.seed)
        except ValueError as err:
            raise HTTPException(400, str(err))
        return state()

    @app.post("/api/demo/restart")
    def restart():
        return {"delivered": rt.restart(), "state": rt.state()}

    @app.post("/api/demo/crash-next")
    def crash_next():
        rt.arm_crash(not rt.crash_next)
        rt.tracer.emit("demo", action="crash_next", armed=rt.crash_next)
        return rt.state()

    @app.post("/api/demo/provider-cancel")
    def provider_cancel():
        try:
            return {"booking": rt.provider_cancel(), "state": rt.state()}
        except ValueError as err:
            raise HTTPException(400, str(err))

    @app.post("/api/demo/advance")
    def advance(seconds: int = 600):
        return {"delivered": rt.advance(seconds), "state": rt.state()}

    @app.get("/api/stream")
    async def stream():
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.add(q)

        async def events():
            try:
                while True:
                    item = await q.get()
                    yield {"event": item["channel"], "data": json.dumps(item, default=str, ensure_ascii=False)}
            finally:
                hub.queues.discard(q)

        return EventSourceResponse(events(), ping=15)

    return app

