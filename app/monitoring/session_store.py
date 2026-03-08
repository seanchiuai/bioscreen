"""In-memory sliding-window session store with TTL expiry.

All public methods are thread-safe via a single :class:`threading.Lock`.
No external dependencies beyond the standard library and the monitoring
schemas defined in :mod:`app.monitoring.schemas`.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from app.monitoring.schemas import SessionEntry, SessionState


class SessionStore:
    """Thread-safe in-memory store for per-session rolling windows.

    Each session holds an ordered list of :class:`~app.monitoring.schemas.SessionEntry`
    objects capped at *window_size*.  Sessions that have been inactive for
    longer than *ttl_seconds* can be purged with :meth:`cleanup_expired`.

    Args:
        window_size: Maximum number of entries retained per session (FIFO).
            When the window is full, the oldest entry is dropped on each
            new addition.
        ttl_seconds: Seconds of inactivity after which a session is
            considered expired and eligible for removal by
            :meth:`cleanup_expired`.
    """

    def __init__(self, window_size: int = 50, ttl_seconds: int = 3600) -> None:
        self._window_size = window_size
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def add_entry(self, session_id: str, entry: SessionEntry) -> SessionState:
        """Append *entry* to the session window, creating the session if new.

        If the window already contains :attr:`window_size` entries the oldest
        entry is discarded first (FIFO trim).  :attr:`~SessionState.last_active`
        is updated to ``entry.timestamp``.

        Args:
            session_id: Opaque session identifier (e.g. client IP hash or
                API key hash).
            entry: The :class:`~app.monitoring.schemas.SessionEntry` to append.

        Returns:
            The updated :class:`~app.monitoring.schemas.SessionState` for the
            session.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            if session_id not in self._sessions:
                state = SessionState(
                    session_id=session_id,
                    entries=[],
                    created_at=now,
                    last_active=now,
                )
                self._sessions[session_id] = state
            else:
                state = self._sessions[session_id]

            # Append then trim oldest entries if over the window limit.
            state.entries.append(entry)
            if len(state.entries) > self._window_size:
                state.entries = state.entries[-self._window_size :]

            # Reflect the timestamp of the most recent query.
            state.last_active = entry.timestamp

            return state

    def cleanup_expired(self) -> int:
        """Remove sessions that have been inactive for longer than *ttl_seconds*.

        The staleness check compares ``datetime.now(timezone.utc)`` against
        each session's :attr:`~SessionState.last_active` timestamp.

        Returns:
            Number of sessions removed.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            expired_ids = [
                sid
                for sid, state in self._sessions.items()
                if (now - state.last_active).total_seconds() > self._ttl_seconds
            ]
            for sid in expired_ids:
                del self._sessions[sid]
        return len(expired_ids)

    def clear(self) -> None:
        """Remove all sessions from the store."""
        with self._lock:
            self._sessions.clear()

    # ------------------------------------------------------------------
    # Read-only methods
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the :class:`~app.monitoring.schemas.SessionState` for a session.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The :class:`~app.monitoring.schemas.SessionState`, or ``None`` if
            the session is not found.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_recent_entries(self, session_id: str, n: int) -> list[SessionEntry]:
        """Return the *n* most-recent entries for a session (newest last).

        If the session has fewer than *n* entries all are returned.  If the
        session does not exist an empty list is returned.

        Args:
            session_id: Opaque session identifier.
            n: Maximum number of recent entries to return.

        Returns:
            A new list containing up to *n* :class:`~app.monitoring.schemas.SessionEntry`
            objects in insertion order.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            # list[-n:] handles n >= len correctly (returns all entries).
            return list(state.entries[-n:]) if n > 0 else []

    def session_count(self) -> int:
        """Return the number of sessions currently held in the store.

        Returns:
            Integer count of active sessions.
        """
        with self._lock:
            return len(self._sessions)
