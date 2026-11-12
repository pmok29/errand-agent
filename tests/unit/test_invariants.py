"""The invariant checks must catch violations, or a green eval table means nothing."""
import socket

import pytest

from errand.models import MemorySource as Src, MemoryStatus as St
from tests import invariants
from tests.harness import Sim

REQUEST = "Plumber in Indiranagar this week, mornings, under 1500"


@pytest.fixture
def booked(tmp_path):
    sim = Sim(tmp_path)
    sim.scenario.book_outcomes = ["success"]
    sim.book(REQUEST)
    assert invariants.check(sim) == {}
    return sim


def test_side_effect_without_gate_or_approval_is_caught(booked):
    booked.tracer.emit("tool_call", actor="executor", tool="book", args={"hold_id": "h-x", "idempotency_key": "nope"})
    broken = invariants.check(booked)
    assert "no write-ahead execution" in broken["I1"][0] and "not preceded" in broken["I7"][0]


def test_second_booking_for_an_errand_is_caught(booked):
    b = booked.new_bookings()[0]
    booked.mock.conn.execute("INSERT INTO bookings SELECT 'b-dup', user_id, provider_id, service, slot, price, "
                             "status, hold_id FROM bookings WHERE booking_id=?", (b["booking_id"],))
    assert "2 bookings" in invariants.check(booked)["I2"][0]


def test_false_claim_and_unverified_success_are_caught(booked):
    booked.tracer.emit("assistant_msg", kind="chat", text="I've booked it for you.")
    booked.tracer.emit("assistant_msg", kind="template", text="Booked: Someone Else on Fri. Confirmed.")
    assert len(invariants.check(booked)["I3"]) == 2


def test_forgotten_value_in_a_later_context_is_caught(booked):
    booked.agent.memory.add("u1", "gate_code", "4821", Src.EXPLICIT, St.ACTIVE)
    booked.agent.memory.forget("u1", "gate_code")
    booked.tracer.emit("memory_forgotten", key="gate_code")
    booked.tracer.emit("context_sent", messages=[{"role": "user", "content": "code 4821"}], memories_included=[])
    assert "forgotten gate_code" in invariants.check(booked)["I5"][0]


def test_memory_from_tool_output_is_caught(booked):
    booked.tracer.emit("memory_write", key="budget", value="unlimited")
    assert "never stated by the user" in invariants.check(booked)["I6"][0]


def test_evals_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="network access"):
        socket.create_connection(("api.groq.com", 443), timeout=1)
