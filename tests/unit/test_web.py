"""Web layer: the API drives the same agent, and system-initiated messages reach the live stream."""
import pytest
from fastapi.testclient import TestClient

from errand.agent.llm.scripted import ReactivePolicy
from errand.web.app import create_app
from errand.web.runtime import Runtime

REQUEST = "Plumber in Indiranagar this week, mornings, under 1500"


@pytest.fixture
def web(tmp_path):
    published = []
    app = create_app(lambda publish: Runtime(tmp_path / "data", None, lambda m: (published.append(m), publish(m)),
                                             llm_factory=lambda on_switch, on_wait: (ReactivePolicy(), "scripted (offline)")),
                     watch=False)
    with TestClient(app) as client:
        client.post("/api/demo/reset", json={"preset": "timeout_booked", "seed": False})
        yield client, app.state.runtime, published


def chat(client, text):
    r = client.post("/api/message", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_index_and_state(web):
    client, _, _ = web
    assert "Errand Agent" in client.get("/").text
    s = client.get("/api/state").json()
    assert s["preset"] == "timeout_booked" and s["llm"] == "scripted (offline)" and s["history"] == []
    assert s["now"].startswith("Tue 10 Nov 09:0")


def test_booking_through_the_confirm_button(web):
    client, rt, published = web
    chat(client, REQUEST)
    card = chat(client, "option 1")["replies"][-1]
    assert card["kind"] == "card" and "CONFIRM" not in card["text"]
    history = client.get("/api/state").json()["history"]
    assert history[-1]["card"] == card["card"]                        # live reply and reload give the same card
    r = client.post("/api/approve", json={"approval_id": card["card"]["approval_id"]})
    out = r.json()
    assert out["replies"][0]["text"].startswith("Booked:") and out["state"]["errand"]["state"] == "DONE"
    after = client.get("/api/state").json()["history"]
    assert after[-2] == {"role": "user", "kind": "approval", "text": "✓ Confirmed"}   # what reload shows
    traced = [m["event"]["type"] for m in published if m["channel"] == "trace"]
    assert {"tool_call", "gate_decision", "context_sent", "approval_consumed"} <= set(traced)


def test_provider_cancel_notification_goes_to_the_stream(web):
    client, rt, published = web
    client.post("/api/demo/reset", json={"preset": "normal", "seed": True})
    booking = client.post("/api/demo/provider-cancel").json()["booking"]
    assert booking["booking_id"] == "b-cln-up"
    pushed = [m["message"] for m in published if m["channel"] == "chat"]
    assert pushed[0]["text"].startswith("Heads up:") and pushed[-1]["kind"] == "card" and pushed[-1]["card"]["code"]
    assert "Heads up" in client.get("/api/state").json()["history"][0]["text"]


def test_crash_then_restart_recovers_and_notifies(web):
    client, rt, published = web
    chat(client, REQUEST)
    code = chat(client, "option 1")["replies"][-1]["card"]["code"]
    assert client.post("/api/demo/crash-next").json()["crash_next"] is True
    assert chat(client, f"CONFIRM {code}")["crashed"] is True
    delivered = client.post("/api/demo/restart").json()["delivered"]
    assert delivered[0]["text"].startswith("Booked:")
    assert len(rt.mock.agent_bookings()) == 1
    assert client.get("/api/state").json()["history"][-1]["text"].startswith("Booked:")


def test_forget_from_the_drawer(web):
    client, rt, _ = web
    client.post("/api/demo/reset", json={"preset": "normal", "seed": True})
    before = {m["key"]: m for m in client.get("/api/memories").json()}
    assert before["preferred_area"]["value"] == "Indiranagar"
    after = {m["key"]: m for m in client.post("/api/memories/forget", json={"key": "preferred_area"}).json()["memories"]}
    assert after["preferred_area"]["status"] == "forgotten" and after["preferred_area"]["value"] == "[forgotten]"


def test_advance_clock_expires_the_card(web):
    client, rt, _ = web
    chat(client, REQUEST)
    code = chat(client, "option 1")["replies"][-1]["card"]["code"]
    client.post("/api/demo/advance?seconds=601")
    assert "expired" in chat(client, f"CONFIRM {code}")["replies"][0]["text"]


def test_bad_requests(web):
    client, _, _ = web
    assert client.post("/api/message", json={"text": "  "}).status_code == 400
    assert client.post("/api/demo/reset", json={"preset": "nope"}).status_code == 400
