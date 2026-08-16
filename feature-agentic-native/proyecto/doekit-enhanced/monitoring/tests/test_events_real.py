"""
Tests para monitoring.events.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from monitoring import EventBus, create_event


class TestEventBus:
    def test_publish_and_retrieve_events(self):
        bus = EventBus()

        bus.publish(create_event("wave.started", {"wave": 1}))
        bus.publish(create_event("wave.finished", {"wave": 1, "status": "ok"}))

        all_events = bus.get_events()
        assert len(all_events) == 2

        started = bus.get_events("wave.started")
        assert len(started) == 1
        assert started[0].payload["wave"] == 1

    def test_subscriber_receives_event(self):
        bus = EventBus()
        captured = []

        def handler(event):
            captured.append(event)

        bus.subscribe("decision.finalized", handler)
        bus.publish(create_event("decision.finalized", {"action": "continue"}))

        assert len(captured) == 1
        assert captured[0].payload["action"] == "continue"
