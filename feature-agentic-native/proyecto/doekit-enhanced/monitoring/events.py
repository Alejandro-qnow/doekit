"""
monitoring.events - Bus de eventos simple para observabilidad del flujo DoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MonitoringEvent:
    """Evento estructurado para auditoria/observabilidad."""

    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EventBus:
    """Pub/sub en memoria para eventos de monitoring."""

    def __init__(self):
        self._events: List[MonitoringEvent] = []
        self._subscribers: Dict[str, List[Callable[[MonitoringEvent], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[MonitoringEvent], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: MonitoringEvent) -> None:
        self._events.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

    def get_events(self, event_type: Optional[str] = None) -> List[MonitoringEvent]:
        if event_type is None:
            return list(self._events)
        return [event for event in self._events if event.event_type == event_type]

    def clear(self) -> None:
        self._events.clear()


def create_event(event_type: str, payload: Optional[Dict[str, Any]] = None, severity: str = "info") -> MonitoringEvent:
    """Helper para crear eventos con timestamp automatico."""
    return MonitoringEvent(event_type=event_type, payload=payload or {}, severity=severity)
