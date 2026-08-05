"""
Memory Systems — Short-term and long-term memory for agent context management.
"""

import json
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from backend.core.config import settings


# ═══════════════════════════════════════════════════════
# SHORT-TERM MEMORY (Conversation Buffer)
# ═══════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Session-scoped conversation buffer memory.
    Stores recent agent interactions and workflow context within a session.
    """

    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.SHORT_TERM_MEMORY_SIZE
        self._sessions: Dict[str, deque] = {}
        self._session_metadata: Dict[str, Dict] = {}

    def get_or_create_session(self, session_id: str) -> deque:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_size)
            self._session_metadata[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0,
                "agents_involved": set(),
            }
        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a message to the session buffer."""
        session = self.get_or_create_session(session_id)
        entry = {
            "id": str(uuid.uuid4())[:8],
            "role": role,
            "content": content[:2000],  # Truncate to avoid bloat
            "agent": agent,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        session.append(entry)
        meta = self._session_metadata[session_id]
        meta["message_count"] += 1
        if agent:
            meta["agents_involved"].add(agent)
        logger.debug(f"[Memory:Short] Added {role} message to session {session_id[:8]}")

    def get_history(self, session_id: str, last_n: int = None) -> List[Dict]:
        """Retrieve session history."""
        session = self.get_or_create_session(session_id)
        history = list(session)
        if last_n:
            history = history[-last_n:]
        return history

    def get_context_string(self, session_id: str, last_n: int = 5) -> str:
        """Get formatted conversation context as a string."""
        history = self.get_history(session_id, last_n)
        if not history:
            return "No prior context in this session."

        lines = []
        for entry in history:
            agent_label = f"[{entry['agent']}]" if entry['agent'] else ""
            content_preview = entry['content'][:300]
            lines.append(f"{entry['role'].upper()} {agent_label}: {content_preview}")

        return "\n\n".join(lines)

    def get_workflow_context(self, session_id: str) -> Dict:
        """Extract structured workflow context from session."""
        history = self.get_history(session_id)
        context = {
            "session_id": session_id,
            "message_count": len(history),
            "agents_seen": list(self._session_metadata.get(session_id, {}).get("agents_involved", set())),
            "last_outputs": {},
        }

        # Extract most recent output from each agent
        for entry in reversed(history):
            agent = entry.get("agent")
            if agent and agent not in context["last_outputs"]:
                context["last_outputs"][agent] = entry["content"][:500]

        return context

    def clear_session(self, session_id: str):
        """Clear a session's memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            del self._session_metadata[session_id]
            logger.info(f"[Memory:Short] Cleared session {session_id[:8]}")

    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        sessions = []
        for sid, meta in self._session_metadata.items():
            agents = list(meta.get("agents_involved", set()))
            sessions.append({
                "session_id": sid,
                "created_at": meta["created_at"],
                "message_count": meta["message_count"],
                "agents_involved": agents,
                "buffer_size": len(self._sessions.get(sid, [])),
            })
        return sessions


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY (Vector Store / Knowledge Base)
# ═══════════════════════════════════════════════════════

class LongTermMemory:
    """
    Persistent knowledge base for cross-session memory.
    Uses ChromaDB for vector-based semantic search (with JSON fallback).
    """

    def __init__(self):
        self.persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._json_store_path = self.persist_dir / "knowledge_store.json"
        self._knowledge_store: List[Dict] = []
        self._chroma_client = None
        self._collection = None
        self._load_json_store()
        self._try_init_chroma()

    def _load_json_store(self):
        """Load knowledge from JSON file (fallback store)."""
        if self._json_store_path.exists():
            try:
                with open(self._json_store_path, "r") as f:
                    self._knowledge_store = json.load(f)
                logger.info(f"[Memory:Long] Loaded {len(self._knowledge_store)} knowledge entries")
            except Exception as e:
                logger.warning(f"[Memory:Long] Could not load store: {e}")
                self._knowledge_store = []

    def _save_json_store(self):
        """Persist knowledge to JSON file."""
        try:
            with open(self._json_store_path, "w") as f:
                json.dump(self._knowledge_store, f, indent=2)
        except Exception as e:
            logger.error(f"[Memory:Long] Could not save store: {e}")

    def _try_init_chroma(self):
        """Try to initialize ChromaDB (gracefully skip if unavailable)."""
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._chroma_client.get_or_create_collection(
                name="agent_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"[Memory:Long] ChromaDB initialized ({self._collection.count()} docs)")
        except Exception as e:
            logger.warning(f"[Memory:Long] ChromaDB unavailable, using JSON fallback: {e}")

    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        workflow_id: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> str:
        """Store knowledge in long-term memory."""
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "content": content[:3000],
            "metadata": {
                **(metadata or {}),
                "workflow_id": workflow_id,
                "agent_role": agent_role,
                "stored_at": datetime.utcnow().isoformat(),
            },
        }

        # Store in JSON (always available)
        self._knowledge_store.append(entry)
        if len(self._knowledge_store) > 1000:  # Cap at 1000 entries
            self._knowledge_store = self._knowledge_store[-1000:]
        self._save_json_store()

        # Try ChromaDB
        if self._collection:
            try:
                self._collection.add(
                    documents=[content[:3000]],
                    metadatas=[entry["metadata"]],
                    ids=[entry_id],
                )
            except Exception as e:
                logger.warning(f"[Memory:Long] ChromaDB store failed: {e}")

        logger.debug(f"[Memory:Long] Stored knowledge entry {entry_id[:8]}")
        return entry_id

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search long-term memory for relevant knowledge."""
        # Try ChromaDB semantic search
        if self._collection and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                return [
                    {"content": doc, "metadata": meta}
                    for doc, meta in zip(docs, metas)
                ]
            except Exception as e:
                logger.warning(f"[Memory:Long] ChromaDB search failed: {e}")

        # Fallback: simple keyword search
        query_words = set(query.lower().split())
        scored = []
        for entry in self._knowledge_store[-100:]:  # Search last 100
            content_words = set(entry["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": e["content"][:500], "metadata": e["metadata"]}
            for _, e in scored[:n_results]
        ]

    def get_recent(self, n: int = 5, agent_role: Optional[str] = None) -> List[Dict]:
        """Get recent knowledge entries, optionally filtered by agent."""
        entries = self._knowledge_store[-50:]  # Last 50
        if agent_role:
            entries = [e for e in entries if e.get("metadata", {}).get("agent_role") == agent_role]
        return entries[-n:]

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "total_entries": len(self._knowledge_store),
            "chroma_available": self._collection is not None,
            "chroma_count": self._collection.count() if self._collection else 0,
            "json_store_size_kb": round(self._json_store_path.stat().st_size / 1024, 1) if self._json_store_path.exists() else 0,
        }


# ═══════════════════════════════════════════════════════
# MEMORY MANAGER (Unified Interface)
# ═══════════════════════════════════════════════════════

class MemoryManager:
    """
    Unified memory interface combining short-term and long-term memory.
    Provides context injection for agent prompts.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        logger.info("MemoryManager initialized")

    def record_agent_output(
        self,
        session_id: str,
        agent_role: str,
        task: str,
        output: str,
        workflow_id: Optional[str] = None,
        persist: bool = True,
    ):
        """Record an agent's output in both short and long-term memory."""
        # Short-term: always
        self.short_term.add_message(
            session_id=session_id,
            role="assistant",
            content=output,
            agent=agent_role,
            metadata={"task": task[:100], "workflow_id": workflow_id},
        )

        # Long-term: if persist flag set (decisions, key findings)
        if persist:
            self.long_term.store(
                content=f"[{agent_role.upper()}] {output}",
                metadata={"task": task[:100]},
                workflow_id=workflow_id,
                agent_role=agent_role,
            )

    def get_context_for_agent(
        self,
        session_id: str,
        agent_role: str,
        query: str,
        include_long_term: bool = True,
    ) -> Dict:
        """Build comprehensive context object for an agent."""
        session_context = self.short_term.get_context_string(session_id)
        workflow_ctx = self.short_term.get_workflow_context(session_id)

        long_term_results = []
        if include_long_term:
            long_term_results = self.long_term.search(query, n_results=2)

        return {
            "session_history": session_context,
            "workflow_context": workflow_ctx,
            "long_term_knowledge": long_term_results,
            "session_id": session_id,
        }

    def get_stats(self) -> Dict:
        """Get unified memory statistics."""
        return {
            "short_term": {
                "active_sessions": len(self.short_term._sessions),
                "sessions": self.short_term.list_sessions(),
            },
            "long_term": self.long_term.get_stats(),
        }


# Singleton instance
memory_manager = MemoryManager()
