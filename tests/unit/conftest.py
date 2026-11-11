from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from errand.agent import approvals
from errand.agent.executor import Executor
from errand.agent.gate import Gate
from errand.agent.state import transition
from errand.agent.store import Store
from errand.clock import FrozenClock
from errand.db import DB
from errand.mocks import tools
from errand.mocks.scenario import ScenarioConfig
from errand.mocks.service import MockProviderService
from errand.models import Action, ErrandKind, ErrandState as S, Request
from errand.tracing.tracer import Tracer
from tests.unit.common import WEEK


@dataclass
class World:
    """Agent + mock wired together on temp DBs. `rebuild()` simulates an agent process restart."""
    dir: Path
    clock: FrozenClock
    scenario: ScenarioConfig
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def __post_init__(self):
        self.mock = MockProviderService(self.dir / "mock.db", self.clock, self.scenario)
        self.rebuild()

    def rebuild(self, faults: set[str] | None = None):
        self.db = DB(self.dir / "agent.db")
        self.tracer = Tracer()
        self.store = Store(self.db, self.clock)
        self.gate = Gate(self.store, self.tracer)
        self.executor = Executor(self.store, self.gate, self.tracer, self.clock, self.call, faults=faults)
        return self

    def call(self, name: str, args: dict):
        self.calls.append((name, dict(args)))
        return tools.call(self.mock, name, args)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    # ---- helpers that do what the orchestrator will do in S5 ----
    def errand(self, kind=ErrandKind.BOOK, **req):
        sid = self.store.create_session("u1")
        request = Request(**({"service": "plumber", "area": "Indiranagar", "date_range": WEEK} | req)) \
            if kind == ErrandKind.BOOK else Request(**req)
        return self.store.create_errand("u1", sid, kind, request)

    def search(self, e, service="plumber", area="Indiranagar"):
        r = self.call("search_providers", {"service": service, "area": area, "date_range": list(WEEK)})
        self.store.record_tool_refs(e.id, "search_providers", r)
        return r

    def hold(self, e, provider_id="p-plb-01", slot_index=0):
        res = self.search(e)
        p = next(p for p in res.data["providers"] if p["provider_id"] == provider_id)
        r = self.call("hold_slot", {"provider_id": provider_id, "slot": p["slots"][slot_index]})
        self.store.record_hold(e.id, provider_id, p["slots"][slot_index], r, p["price"])
        if e.state == S.GATHERING:
            transition(self.store, self.tracer, e, S.COMPARING, "test")
        transition(self.store, self.tracer, e, S.HOLDING, "test")
        return r

    def approve(self, e, action=Action.BOOK, **kw):
        draft = approvals.build_draft(self.store, e, action, **kw)
        a, card = approvals.create(self.store, self.tracer, e, draft, 600)
        transition(self.store, self.tracer, e, S.AWAITING_APPROVAL, "test")
        return a, card

    def booked_and_approved(self, provider_id="p-plb-01"):
        e = self.errand()
        self.hold(e, provider_id)
        a, card = self.approve(e)
        res = approvals.consume(self.store, self.tracer, e, a.code)
        assert res.status == "ok"
        return self.store.get_errand(e.id), res.execution

    def booking(self, booking_id: str) -> dict:
        return next(b for b in self.mock.ledger() if b["booking_id"] == booking_id)

    def bookings_for_hold(self, hold_id: str) -> int:
        return sum(1 for b in self.mock.ledger() if b["hold_id"] == hold_id)


@pytest.fixture
def world_factory(tmp_path):
    def make(**cfg) -> World:
        return World(tmp_path, FrozenClock(), ScenarioConfig(**cfg))
    return make
