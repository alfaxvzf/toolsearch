# vector_tool_router_gigachat_demo.py
#
# pip install -U deepagents langchain langchain-core langchain-gigachat
#
# env example:
#   export GIGACHAT_CREDENTIALS="..."
#   export GIGACHAT_SCOPE="GIGACHAT_API_PERS"
#   export GIGACHAT_BASE_URL="https://your-gigachat-api-url"
#   export GIGACHAT_MODEL="GigaChat-2-Max"
#   export GIGACHAT_EMBEDDING_MODEL="Embeddings"

from __future__ import annotations

import json
import math
import os

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_gigachat import GigaChat, GigaChatEmbeddings


# ----------------------------
# 1. GigaChat embeddings adapter
# ----------------------------

class CallableGigaEmbeddings(GigaChatEmbeddings):
    """
    Compatibility adapter.

    Some LangChain/vectorstore integrations expect embedding_function(texts)
    in addition to standard embed_documents/embed_query.
    """

    def __call__(self, input):
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(list(input))


# ----------------------------
# 2. Business tools
# ----------------------------

@tool
def search_docs(query: str) -> str:
    """Search documentation, knowledge base, API docs, incidents, TLS errors, certificates."""
    return (
        "Docs result: TLS hostname mismatch usually means the certificate SAN "
        "does not contain the requested hostname. Check SAN, DNS name, chain, "
        "truststore, and mTLS client certificate."
    )


@tool
def make_checklist(service: str, problem: str) -> str:
    """Create a short troubleshooting checklist or incident investigation plan."""
    return (
        f"Checklist for {service} / {problem}:\n"
        "1. Check application and gateway logs.\n"
        "2. Check certificate SAN.\n"
        "3. Check certificate chain.\n"
        "4. Check gateway truststore.\n"
        "5. Check mTLS client certificate.\n"
        "6. Check route/upstream config.\n"
        "7. Then check auth token and scope."
    )


TOOLS = [search_docs, make_checklist]
TOOL_NAMES = {t.name for t in TOOLS}


# ----------------------------
# 3. Tool cards for vector search
# ----------------------------

def tool_to_document(tool) -> Document:
    schema = ""
    if getattr(tool, "args_schema", None) is not None:
        schema = json.dumps(
            tool.args_schema.model_json_schema(),
            ensure_ascii=False,
        )

    return Document(
        page_content=f"""
Tool name: {tool.name}

Description:
{tool.description}

Input schema:
{schema}
""".strip(),
        metadata={"tool_name": tool.name},
    )


# ----------------------------
# 4. Very simple in-memory vectorstore
# ----------------------------

class SimpleVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.items = []

    def add_documents(self, docs: list[Document]) -> None:
        vectors = self.embeddings.embed_documents([doc.page_content for doc in docs])
        self.items.extend(zip(docs, vectors))

    def similarity_search(self, query: str, k: int = 1) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)

        scored = [
            (self.cosine(query_vector, vector), doc)
            for doc, vector in self.items
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[:k]]

    def as_retriever(self, k: int = 1):
        return SimpleRetriever(self, k)

    def cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class SimpleRetriever:
    def __init__(self, vectorstore: SimpleVectorStore, k: int = 1):
        self.vectorstore = vectorstore
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)


# ----------------------------
# 5. Middleware: vector-based tool routing
# ----------------------------

def last_user_text(messages) -> str:
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in {"human", "user"}:
            return str(getattr(msg, "content", ""))
    return ""


class VectorToolRouter(AgentMiddleware):
    def __init__(self, tool_retriever, tool_names: set[str], fallback_tool: str):
        self.tool_retriever = tool_retriever
        self.tool_names = tool_names
        self.fallback_tool = fallback_tool

    def wrap_model_call(self, request, handler):
        query = last_user_text(request.messages)

        selected = {
            doc.metadata["tool_name"]
            for doc in self.tool_retriever.invoke(query)
            if doc.metadata.get("tool_name") in self.tool_names
        } or {self.fallback_tool}

        visible_tools = [
            tool for tool in request.tools
            if tool.name not in self.tool_names or tool.name in selected
        ]

        print("[tool-router] query:", query)
        print("[tool-router] selected:", sorted(selected))
        print("[tool-router] visible:", [tool.name for tool in visible_tools])

        return handler(request.override(tools=visible_tools))


# ----------------------------
# 6. Build model, embeddings, index, agent
# ----------------------------

def build_gigachat_common_kwargs() -> dict:
    return {
        "credentials": os.environ["GIGACHAT_CREDENTIALS"],
        "scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        "base_url": os.environ.get("GIGACHAT_BASE_URL"),
        "verify_ssl_certs": os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true",
    }


def main():
    common_kwargs = build_gigachat_common_kwargs()

    llm = GigaChat(
        **common_kwargs,
        model=os.environ.get("GIGACHAT_MODEL", "GigaChat-2-Max"),
        temperature=0,
        function_ranker={"enabled": False},
        allow_any_tool_choice_fallback=True,
    )

    embeddings = CallableGigaEmbeddings(
        **common_kwargs,
        model=os.environ.get("GIGACHAT_EMBEDDING_MODEL", "Embeddings"),
    )

    tool_docs = [tool_to_document(tool) for tool in TOOLS]

    vectorstore = SimpleVectorStore(embeddings)
    vectorstore.add_documents(tool_docs)

    tool_retriever = vectorstore.as_retriever(k=1)

    agent = create_deep_agent(
        model=llm,
        tools=TOOLS,
        middleware=[
            VectorToolRouter(
                tool_retriever=tool_retriever,
                tool_names=TOOL_NAMES,
                fallback_tool="search_docs",
            )
        ],
        system_prompt="Use available tools when useful. Answer briefly.",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Составь план проверки Sber API после TLS hostname mismatch.",
                }
            ]
        },
        config={
            "configurable": {"thread_id": "vector-tool-router-demo-1"},
            "tags": ["gigachat", "deep-agent", "vector-tool-routing"],
        },
    )

    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()