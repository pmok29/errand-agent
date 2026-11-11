"""Argument validation and dispatch of the fixed-contract tools to the mock service."""
from __future__ import annotations

from typing import Any

from errand.contract import CONTRACT, OPTIONAL
from errand.mocks.service import MockProviderService
from errand.models import ToolResult


def call(service: MockProviderService, name: str, args: dict[str, Any]) -> ToolResult:
    """Validate arguments against the contract and invoke the tool. May raise ToolTimeout."""
    if name not in CONTRACT:
        return ToolResult.fail("unknown_tool", message=name)
    allowed = set(CONTRACT[name]) | set(OPTIONAL.get(name, ()))
    missing = [a for a in CONTRACT[name] if a not in args]
    extra = [a for a in args if a not in allowed]
    if missing or extra:
        return ToolResult.fail("invalid_arguments", message=f"missing={missing} unexpected={extra}")
    return getattr(service, name)(**args)
