import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentsentinel.models import RiskCategory, RiskFlag, RiskLevel


MEMORY_BACKENDS = {
    "MemorySaver":          "langgraph.memory_saver",
    "SqliteSaver":          "langgraph.sqlite_checkpointer",
    "PostgresSaver":        "langgraph.postgres_checkpointer",
    "AsyncSqliteSaver":     "langgraph.sqlite_checkpointer",
    "AsyncPostgresSaver":   "langgraph.postgres_checkpointer",
    "RedisSaver":           "langgraph.redis_checkpointer",
    "RedisChatMessageHistory": "langchain.redis_chat_history",
    "ConversationBufferMemory": "langchain.buffer_memory",
    "ConversationSummaryMemory": "langchain.summary_memory",
    "Chroma":              "vector_store.chroma",
    "FAISS":               "vector_store.faiss",
    "Pinecone":            "vector_store.pinecone",
    "Weaviate":            "vector_store.weaviate",
    "Qdrant":              "vector_store.qdrant",
}

TTL_HINTS = ["ttl", "expir", "max_age", "retention", "evict"]
USER_DATA_HINTS = ["user_id", "user_email", "user.email", "user_name", "pii", "phi", "personal_data"]
SCOPE_HINTS = ["thread_id", "session_id", "user_id", "namespace", "scope", "partition"]


class MemoryAnalysis(BaseModel):
    has_memory: bool = False
    memory_type: Optional[str] = None
    memory_risks: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)


def _detect_from_object(agent: Any) -> Optional[str]:
    if agent is None:
        return None
    cp = getattr(agent, "checkpointer", None)
    if cp is not None:
        return type(cp).__name__
    store = getattr(agent, "store", None)
    if store is not None:
        return type(store).__name__
    return None


def _detect_from_source(source: str) -> Optional[str]:
    for backend in MEMORY_BACKENDS:
        if re.search(rf"\b{re.escape(backend)}\b", source):
            return backend
    return None


def analyze_memory(agent: Any = None, source_code: Optional[str] = None) -> MemoryAnalysis:
    """Static memory/state analysis. Detects backend, TTL, scope separation."""
    backend = _detect_from_object(agent)
    if backend is None and source_code:
        backend = _detect_from_source(source_code)

    if backend is None:
        return MemoryAnalysis(has_memory=False)

    memory_type = MEMORY_BACKENDS.get(backend, backend.lower())
    risks: list[str] = []

    if source_code:
        # String-based checks are only meaningful when source is available.
        # Skipping them for runtime agents avoids false-alarm flags.
        lowered = source_code.lower()
        has_ttl = any(h in lowered for h in TTL_HINTS)
        stores_user_data = any(h in lowered for h in USER_DATA_HINTS)
        has_scope = any(h in lowered for h in SCOPE_HINTS)

        if not has_ttl:
            risks.append("No TTL/expiry configuration detected — unbounded memory growth + retention risk.")
        if stores_user_data and not has_scope:
            risks.append("Stores user-identifying data without visible scope separation — cross-user leak risk.")
        if stores_user_data:
            risks.append("Stores user data — review HIPAA / SOC 2 / GDPR applicability.")
        if not has_scope:
            risks.append("No thread_id/session_id/namespace scoping detected — sessions may share state.")

    flags: list[RiskFlag] = []
    if risks:
        flags.append(RiskFlag(
            category=RiskCategory.MEMORY_RISK,
            description=f"Memory backend '{backend}' has {len(risks)} configuration issue(s).",
            location=f"memory:{memory_type}",
            severity=RiskLevel.MEDIUM,
            suggestion="Configure a TTL, namespace memory by user/session, and document what user data is persisted.",
        ))

    return MemoryAnalysis(
        has_memory=True,
        memory_type=memory_type,
        memory_risks=risks,
        risk_flags=flags,
    )
