# two_phase_deepagents_poc.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Any, Callable, Literal, Sequence

from typing_extensions import NotRequired

from deepagents import create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_gigachat import GigaChat, GigaEmbeddings


# ---------------------------------------------------------------------
# Model / embeddings
# ---------------------------------------------------------------------

def build_gigachat() -> GigaChat:
    return GigaChat(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS", ""),
        base_url=os.environ.get("GIGACHAT_API_BASE", ""),
        scope=os.environ.get("GIGACHAT_SCOPE", ""),
        model=os.environ.get("GIGACHAT_MODEL", ""),
        temperature=float(os.environ.get("GIGACHAT_TEMPERATURE", "0")),
        timeout=int(os.environ.get("GIGACHAT_TIMEOUT", "120")),
        verify_ssl_certs=False,
        profanity_check=False,
    )


def build_embeddings() -> GigaEmbeddings:
    return GigaEmbeddings(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS", ""),
        base_url=os.environ.get("GIGACHAT_API_BASE", ""),
        scope=os.environ.get("GIGACHAT_SCOPE", ""),
        model=os.environ.get("GIGACHAT_EMBEDDING_MODEL", ""),
        verify_ssl_certs=False,
    )


# ---------------------------------------------------------------------
# Business tools: self-sufficient, no knowledge vectorstore
# ---------------------------------------------------------------------

def build_business_tools() -> list[BaseTool]:
    @tool
    def search(query: str) -> str:
        """
        Search an external or internal system for relevant information.
        Use this when the answer requires data lookup.
        """
        # PoC stub. In real code this may call DB, API, MCP, retriever, etc.
        return (
            f"Search results for: {query}\n"
            "- No real backend is connected in this PoC.\n"
            "- Replace this body with your own search implementation."
        )

    @tool
    def checklist(topic: str) -> str:
        """
        Produce a practical diagnostic checklist.
        Use this when the user needs structured troubleshooting steps or an action plan.
        """
        return (
            f"Checklist for: {topic}\n"
            "1. Clarify the exact symptom and expected behavior.\n"
            "2. Check the latest deployment, config, credentials, and routes.\n"
            "3. Check readiness, health endpoints, and dependency availability.\n"
            "4. Inspect logs around the first failure timestamp.\n"
            "5. Reproduce with the smallest possible request.\n"
            "6. Isolate whether the issue is client-side, gateway-side, or upstream.\n"
            "7. Record findings, owner, and next action."
        )

    return [search, checklist]


# ---------------------------------------------------------------------
# Runtime vector index over business tools only
# ---------------------------------------------------------------------

class VectorToolSelector:
    def __init__(
        self,
        embeddings: GigaEmbeddings,
        business_tools: Sequence[BaseTool],
    ):
        self.tools_by_name = {t.name: t for t in business_tools}
        self.vectorstore = InMemoryVectorStore(embedding=embeddings)

        self.vectorstore.add_documents(
            [
                Document(
                    page_content=self._tool_text(t),
                    metadata={"tool_name": t.name},
                )
                for t in business_tools
            ]
        )

    def select(
        self,
        query: str,
        candidate_tools: Sequence[BaseTool],
        k: int,
    ) -> list[BaseTool]:
        candidate_names = {t.name for t in candidate_tools}

        docs = self.vectorstore.similarity_search(
            query=query,
            k=max(k, len(self.tools_by_name)),
        )

        selected = [
            self.tools_by_name[name]
            for doc in docs
            if (name := doc.metadata.get("tool_name")) in candidate_names
            and name in self.tools_by_name
        ]

        return selected[:k] or list(candidate_tools)[:k]

    def _tool_text(self, tool_obj: BaseTool) -> str:
        return (
            f"name: {tool_obj.name}\n"
            f"description: {tool_obj.description or ''}\n"
            f"args: {json.dumps(tool_obj.args, ensure_ascii=False, default=str)}"
        )


# ---------------------------------------------------------------------
# Two-phase middleware
# ---------------------------------------------------------------------

HARNESS_TOOL_NAMES = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "task",
}


class TwoPhaseState(AgentState):
    tool_phase: NotRequired[Literal["business", "harness"]]
    model_call_count: NotRequired[int]

class TwoPhaseToolRouter(AgentMiddleware[TwoPhaseState]):
    """
    business phase:
      expose only vector-selected business tools

    harness phase:
      expose only DeepAgents harness tools

    The point is to avoid showing weak models business tools and harness tools
    at the same time.
    """

    state_schema = TwoPhaseState

    PHASE_INSTRUCTIONS = {
        "business": (
            "BUSINESS PHASE.\n"
            "Only selected business tools are visible now.\n"
            "Use search for lookup. Use checklist for structured diagnostic steps. "
            "If no tool is needed, answer directly."
        ),
        "harness": (
            "HARNESS PHASE.\n"
            "Only harness tools are visible now.\n"
            "Use at most one harness tool if useful for todos, notes, files, or delegation. "
            "Do not use business tools here.\n"
            "If no harness action is useful, answer exactly: NO_HARNESS_ACTION."
        ),
    }

    def __init__(
        self,
        selector: VectorToolSelector,
        business_tool_names: set[str],
        max_business_tools: int = 2,
        max_model_calls: int = 12,
    ):
        self.selector = selector
        self.business_tool_names = business_tool_names
        self.max_business_tools = max_business_tools
        self.max_model_calls = max_model_calls

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: TwoPhaseState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0) + 1

        if count > self.max_model_calls:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Stopped: two-phase tool loop exceeded "
                            f"{self.max_model_calls} model calls."
                        )
                    )
                ],
                "jump_to": "end",
            }

        return {
            "model_call_count": count,
            "tool_phase": state.get("tool_phase", "business"),
        }

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        phase = request.state.get("tool_phase", "business")
        all_tools = list(request.tools or [])

        business_tools = [
            tool for tool in all_tools
            if tool.name in self.business_tool_names
        ]

        harness_tools = [
            tool for tool in all_tools
            if tool.name in HARNESS_TOOL_NAMES
        ]

        phase_tools = {
            "business": lambda: self.selector.select(
                query=self._last_user_text(request),
                candidate_tools=business_tools,
                k=self.max_business_tools,
            ),
            "harness": lambda: harness_tools,
        }

        routed = request.override(tools=phase_tools[phase]())
        routed = self._append_system_instruction(
            routed,
            self.PHASE_INSTRUCTIONS[phase],
        )

        return handler(routed)

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: TwoPhaseState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        phase = state.get("tool_phase", "business")
        messages = state.get("messages", [])

        did_call_tool = bool(
            messages
            and (getattr(messages[-1], "tool_calls", None) or [])
        )

        if phase == "business" and did_call_tool:
            return {"tool_phase": "harness"}

        if phase == "harness":
            update: dict[str, Any] = {"tool_phase": "business"}

            if not did_call_tool:
                update["jump_to"] = "model"

            return update

        return None

    def _last_user_text(self, request: ModelRequest) -> str:
        for message in reversed(request.messages):
            if getattr(message, "type", None) == "human":
                content = getattr(message, "content", "")
                return content if isinstance(content, str) else str(content)

        return ""

    def _append_system_instruction(
        self,
        request: ModelRequest,
        text: str,
    ) -> ModelRequest:
        if request.system_message is None:
            return request.override(
                system_message=SystemMessage(content=text)
            )

        content = list(request.system_message.content_blocks)
        content.append({"type": "text", "text": text})

        return request.override(
            system_message=SystemMessage(content=content)
        )
# ---------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------

def build_agent(
    llm: GigaChat,
    embeddings: GigaEmbeddings,
) -> Any:
    business_tools = build_business_tools()

    selector = VectorToolSelector(
        embeddings=embeddings,
        business_tools=business_tools,
    )

    router = TwoPhaseToolRouter(
        selector=selector,
        business_tool_names={t.name for t in business_tools},
        max_business_tools=2,
        max_model_calls=int(os.environ.get("MAX_MODEL_CALLS", "12")),
    )

    return create_deep_agent(
        model=llm,
        tools=business_tools,
        middleware=[router],
        system_prompt=(
            "You are a technical assistant. "
            "The runtime alternates between business-tool and harness-tool phases."
        ),
    )


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def last_useful_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) != "ai":
            continue

        content = getattr(message, "content", "")

        if content and content != "NO_HARNESS_ACTION":
            return content if isinstance(content, str) else str(content)

    return str(result)


def main() -> None:
    agent = build_agent(
        llm=build_gigachat(),
        embeddings=build_embeddings(),
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "У нас после деплоя не работает SBER API. "
                        "Дай план диагностики."
                    )
                )
            ],
            "tool_phase": "business",
            "model_call_count": 0,
        }
    )

    print(last_useful_ai_text(result))


if __name__ == "__main__":
    main()