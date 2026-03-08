"""Session/behavioral monitoring for convergent optimization detection.

Exports module-level singletons for use by the API layer:

* :data:`default_store` — shared :class:`~app.monitoring.session_store.SessionStore`
  with a 50-entry rolling window and 1-hour session TTL.
* :data:`default_analyzer` — shared :class:`~app.monitoring.analyzer.SessionAnalyzer`
  using all default detection thresholds.
"""

from app.monitoring.analyzer import SessionAnalyzer
from app.monitoring.session_store import SessionStore

__all__ = ["default_store", "default_analyzer"]

default_store: SessionStore = SessionStore(window_size=50, ttl_seconds=3600)
default_analyzer: SessionAnalyzer = SessionAnalyzer()
