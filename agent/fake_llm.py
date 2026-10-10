"""Deterministic offline chat model for tests/demo (FAKE_LLM=1). Needs no network.

It reads the stage marker in the system prompt ([PLAN], [INVESTIGATE], [SUMMARIZE]) and, for the
investigate stage, emits sensible tool calls: list_customers, then reconcile_customer for the
customers named/numbered in the user's request (or for everybody).
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeBillingLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-billing-llm"

    def bind_tools(self, tools: Any, **kwargs: Any):  # tools are chosen by the script below
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        system = system if isinstance(system, str) else ""
        if system.startswith("[PLAN]"):
            msg = AIMessage(content="Plan: list customers, reconcile billing plan vs invoices for the "
                                    "requested scope, quantify leakage, propose corrective actions for approval.")
        elif system.startswith("[INVESTIGATE]"):
            msg = self._investigate(messages)
        else:
            msg = AIMessage(content="Investigation complete. Findings and recommended corrective actions "
                                    "are listed below; every write needs approver sign-off.")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _investigate(self, messages: list[BaseMessage]) -> AIMessage:
        last_h = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        request = str(messages[last_h].content).lower()
        tools = [m for m in messages[last_h + 1:] if isinstance(m, ToolMessage)]
        listed = next((m for m in tools if m.name == "list_customers"), None)
        if listed is None:
            return AIMessage(content="", tool_calls=[
                {"name": "list_customers", "args": {}, "id": "call_list", "type": "tool_call"}])
        if any(m.name == "reconcile_customer" for m in tools):
            return AIMessage(content="Reconciliation finished for the requested customers.")
        customers = json.loads(listed.content)
        ids = {int(x) for x in re.findall(r"customer\s*(?:#|id\s*)?(\d+)", request)}
        ids |= {c["id"] for c in customers if c["name"].lower() in request}
        if not ids:
            ids = {c["id"] for c in customers}
        return AIMessage(content="", tool_calls=[
            {"name": "reconcile_customer", "args": {"customer_id": i}, "id": f"call_rec_{i}",
             "type": "tool_call"} for i in sorted(ids)])
