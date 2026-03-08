"""Tests for app/monitoring/session_store.py.

Follows the same conventions as tests/test_pipeline.py and tests/test_schemas.py:
plain pytest functions, direct assert statements, no test classes, no external
fixtures.  All helpers are module-level factory functions.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from app.monitoring.schemas import EMBEDDING_DIM, SessionEntry, SessionState
from app.monitoring.session_store import SessionStore

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

VALID_HASH = "b" * 64  # 64 lowercase hex chars
VALID_EMBEDDING = [0.0] * EMBEDDING_DIM
NOW = datetime.now(timezone.utc)


def _make_entry(
    *,
    sequence_hash: str = VALID_HASH,
    risk_score: float = 0.3,
    sequence_length: int = 50,
    timestamp: datetime | None = None,
) -> SessionEntry:
    """Return a valid SessionEntry with sensible defaults."""
    return SessionEntry(
        sequence_hash=sequence_hash,
        embedding=VALID_EMBEDDING,
        timestamp=timestamp or NOW,
        risk_score=risk_score,
        sequence_length=sequence_length,
    )


def _fresh_store(**kwargs) -> SessionStore:
    """Return a new, empty SessionStore (default or custom kwargs)."""
    return SessionStore(**kwargs)


# ---------------------------------------------------------------------------
# add_entry — session creation
# ---------------------------------------------------------------------------


def test_add_entry_creates_new_session():
    store = _fresh_store()
    entry = _make_entry()
    state = store.add_entry("s1", entry)

    assert state.session_id == "s1"
    assert len(state.entries) == 1
    assert state.entries[0] is entry


def test_add_entry_returns_session_state():
    store = _fresh_store()
    result = store.add_entry("s1", _make_entry())
    assert isinstance(result, SessionState)


def test_add_entry_sets_created_at():
    store = _fresh_store()
    before = datetime.now(timezone.utc)
    store.add_entry("s1", _make_entry())
    after = datetime.now(timezone.utc)

    state = store.get_session("s1")
    assert before <= state.created_at <= after


def test_add_entry_updates_last_active_to_entry_timestamp():
    store = _fresh_store()
    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    entry = _make_entry(timestamp=ts)
    store.add_entry("s1", entry)

    state = store.get_session("s1")
    assert state.last_active == ts


# ---------------------------------------------------------------------------
# add_entry — appending to existing session
# ---------------------------------------------------------------------------


def test_add_entry_appends_to_existing_session():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    store.add_entry("s1", _make_entry())
    store.add_entry("s1", _make_entry())

    state = store.get_session("s1")
    assert len(state.entries) == 3


def test_add_entry_does_not_reset_created_at_on_subsequent_calls():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    state_after_first = store.get_session("s1")
    created_at_first = state_after_first.created_at

    store.add_entry("s1", _make_entry())
    state_after_second = store.get_session("s1")
    assert state_after_second.created_at == created_at_first


# ---------------------------------------------------------------------------
# Window trimming
# ---------------------------------------------------------------------------


def test_window_trimming_keeps_only_newest_entries():
    window_size = 5
    store = _fresh_store(window_size=window_size)
    extra = 10
    total = window_size + extra

    entries = [_make_entry(sequence_length=i + 1) for i in range(total)]
    for e in entries:
        store.add_entry("s1", e)

    state = store.get_session("s1")
    assert len(state.entries) == window_size

    # The retained entries must be the NEWEST (last window_size added).
    expected = entries[-window_size:]
    assert state.entries == expected


def test_window_trimming_exact_boundary():
    window_size = 3
    store = _fresh_store(window_size=window_size)

    for i in range(window_size):
        store.add_entry("s1", _make_entry(sequence_length=i + 1))

    state = store.get_session("s1")
    assert len(state.entries) == window_size  # no trim yet


def test_window_trimming_one_over_boundary():
    window_size = 3
    store = _fresh_store(window_size=window_size)
    entries = [_make_entry(sequence_length=i + 1) for i in range(window_size + 1)]
    for e in entries:
        store.add_entry("s1", e)

    state = store.get_session("s1")
    assert len(state.entries) == window_size
    assert state.entries[0].sequence_length == 2  # oldest (length=1) was dropped


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_returns_none_for_unknown_id():
    store = _fresh_store()
    assert store.get_session("nonexistent") is None


def test_get_session_returns_correct_state():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    store.add_entry("s2", _make_entry())

    s1 = store.get_session("s1")
    s2 = store.get_session("s2")

    assert s1.session_id == "s1"
    assert s2.session_id == "s2"


# ---------------------------------------------------------------------------
# get_recent_entries
# ---------------------------------------------------------------------------


def test_get_recent_entries_returns_last_n():
    store = _fresh_store()
    entries = [_make_entry(sequence_length=i + 1) for i in range(10)]
    for e in entries:
        store.add_entry("s1", e)

    recent = store.get_recent_entries("s1", 3)
    assert len(recent) == 3
    assert recent == entries[-3:]


def test_get_recent_entries_n_larger_than_entries_returns_all():
    store = _fresh_store()
    entries = [_make_entry(sequence_length=i + 1) for i in range(4)]
    for e in entries:
        store.add_entry("s1", e)

    recent = store.get_recent_entries("s1", 100)
    assert len(recent) == 4
    assert recent == entries


def test_get_recent_entries_n_equals_entry_count():
    store = _fresh_store()
    entries = [_make_entry(sequence_length=i + 1) for i in range(5)]
    for e in entries:
        store.add_entry("s1", e)

    recent = store.get_recent_entries("s1", 5)
    assert recent == entries


def test_get_recent_entries_returns_empty_for_unknown_session():
    store = _fresh_store()
    assert store.get_recent_entries("ghost", 10) == []


def test_get_recent_entries_n_zero_returns_empty():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    assert store.get_recent_entries("s1", 0) == []


def test_get_recent_entries_returns_new_list_not_internal_ref():
    """Mutating the returned list must not affect the stored session."""
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    recent = store.get_recent_entries("s1", 10)
    recent.clear()

    state = store.get_session("s1")
    assert len(state.entries) == 1


# ---------------------------------------------------------------------------
# cleanup_expired
# ---------------------------------------------------------------------------


def test_cleanup_expired_removes_stale_sessions():
    # Use a past timestamp so the session is immediately expired.
    store = _fresh_store(ttl_seconds=60)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)  # 2 min ago
    store.add_entry("stale", _make_entry(timestamp=stale_ts))

    removed = store.cleanup_expired()

    assert removed == 1
    assert store.get_session("stale") is None


def test_cleanup_expired_leaves_active_sessions_untouched():
    store = _fresh_store(ttl_seconds=3600)
    # Fresh entry with current timestamp — not expired.
    store.add_entry("active", _make_entry(timestamp=datetime.now(timezone.utc)))

    removed = store.cleanup_expired()

    assert removed == 0
    assert store.get_session("active") is not None


def test_cleanup_expired_mixed_sessions():
    store = _fresh_store(ttl_seconds=60)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)

    store.add_entry("stale1", _make_entry(timestamp=stale_ts))
    store.add_entry("stale2", _make_entry(timestamp=stale_ts))
    store.add_entry("active", _make_entry(timestamp=datetime.now(timezone.utc)))

    removed = store.cleanup_expired()

    assert removed == 2
    assert store.get_session("stale1") is None
    assert store.get_session("stale2") is None
    assert store.get_session("active") is not None


def test_cleanup_expired_returns_zero_when_nothing_to_remove():
    store = _fresh_store(ttl_seconds=3600)
    removed = store.cleanup_expired()
    assert removed == 0


# ---------------------------------------------------------------------------
# session_count
# ---------------------------------------------------------------------------


def test_session_count_empty_store():
    store = _fresh_store()
    assert store.session_count() == 0


def test_session_count_after_adds():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    store.add_entry("s2", _make_entry())
    store.add_entry("s3", _make_entry())
    assert store.session_count() == 3


def test_session_count_same_session_id_does_not_increment():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    store.add_entry("s1", _make_entry())
    store.add_entry("s1", _make_entry())
    assert store.session_count() == 1


def test_session_count_decrements_after_cleanup():
    store = _fresh_store(ttl_seconds=60)
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
    store.add_entry("stale", _make_entry(timestamp=stale_ts))
    store.add_entry("active", _make_entry(timestamp=datetime.now(timezone.utc)))

    assert store.session_count() == 2
    store.cleanup_expired()
    assert store.session_count() == 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_store():
    store = _fresh_store()
    store.add_entry("s1", _make_entry())
    store.add_entry("s2", _make_entry())
    store.clear()

    assert store.session_count() == 0
    assert store.get_session("s1") is None
    assert store.get_session("s2") is None


def test_clear_on_empty_store_is_safe():
    store = _fresh_store()
    store.clear()  # must not raise
    assert store.session_count() == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_add_entry_same_session():
    """10 threads each add 1 entry to the same session; no race condition."""
    store = _fresh_store(window_size=100)
    session_id = "concurrent"
    num_threads = 10
    errors: list[Exception] = []

    def worker() -> None:
        try:
            store.add_entry(session_id, _make_entry())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Threads raised exceptions: {errors}"
    state = store.get_session(session_id)
    assert state is not None
    assert len(state.entries) == num_threads


def test_thread_safety_concurrent_add_entry_different_sessions():
    """10 threads each create and populate their own session; no data corruption."""
    store = _fresh_store(window_size=100)
    num_threads = 10
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            sid = f"session-{idx}"
            for _ in range(5):
                store.add_entry(sid, _make_entry())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Threads raised exceptions: {errors}"
    assert store.session_count() == num_threads
    for i in range(num_threads):
        state = store.get_session(f"session-{i}")
        assert state is not None
        assert len(state.entries) == 5


def test_thread_safety_window_trim_under_concurrency():
    """Window trimming is correct when many threads add entries concurrently."""
    window_size = 10
    store = _fresh_store(window_size=window_size)
    session_id = "trim-test"
    num_threads = 20  # more than window_size
    errors: list[Exception] = []

    def worker() -> None:
        try:
            store.add_entry(session_id, _make_entry())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Threads raised exceptions: {errors}"
    state = store.get_session(session_id)
    # After trimming, exactly window_size entries must remain.
    assert len(state.entries) == window_size
