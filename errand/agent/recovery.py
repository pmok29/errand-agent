"""Startup recovery: finish any side effect that was in flight when the process died."""
from __future__ import annotations

from errand.agent.executor import Executor
from errand.agent.templates import Outcome
from errand.tracing.tracer import Tracer


def recover(executor: Executor, tracer: Tracer) -> list[Outcome]:
    """Resume every open execution. The Agent tells the users (Agent.recover)."""
    outcomes = []
    for exe in executor.store.open_executions():
        tracer.emit("recovery_resume", execution_id=exe.id, errand_id=exe.errand_id, action=exe.action,
                    state=exe.state, idem_key=exe.idem_key)
        outcomes.append(executor.resume(exe))
    return outcomes
