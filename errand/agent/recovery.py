"""Startup recovery: finish any side effect that was in flight when the process died."""
from __future__ import annotations

from errand.agent.executor import Executor
from errand.agent.templates import Outcome, render
from errand.tracing.tracer import Tracer


def recover(executor: Executor, tracer: Tracer) -> list[Outcome]:
    outcomes = []
    for exe in executor.store.open_executions():
        tracer.emit("recovery_resume", execution_id=exe.id, errand_id=exe.errand_id, action=exe.action,
                    state=exe.state, idem_key=exe.idem_key)
        outcome = executor.resume(exe)
        errand = executor.store.get_errand(exe.errand_id)
        executor.store.add_notification(errand.session_id, {"kind": "template", "outcome": outcome.kind,
                                                            "text": render(outcome), "card": outcome.card})
        outcomes.append(outcome)
    return outcomes
