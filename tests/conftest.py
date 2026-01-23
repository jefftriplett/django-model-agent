"""
pytest configuration and fixtures for django-model-agent tests.
"""

from __future__ import annotations

import pytest

from tests.models import Place

from django_model_agent import ModelAgent


@pytest.fixture
def place(db):
    """Create a test Place instance."""
    return Place.objects.create(
        name="Test Restaurant",
        slug="test-restaurant",
        address="123 Main St",
        locality="Lawrence",
        region="KS",
        phone="785-555-1234",
        website="https://example.com",
        description="A test restaurant",
        state=Place.STATE_PUBLIC,
        delivery=True,
        takeout=True,
        dinein=True,
        doordash_url="https://doordash.com/test",
        grubhub_url="https://grubhub.com/test",
    )


@pytest.fixture
def draft_place(db):
    """Create a draft Place instance."""
    return Place.objects.create(
        name="Draft Restaurant",
        slug="draft-restaurant",
        state=Place.STATE_DRAFT,
    )


@pytest.fixture
def simple_agent_class():
    """Create a simple ModelAgent subclass for testing."""

    class SimpleAgent(ModelAgent):
        model = Place
        fields = ["name", "address", "phone"]
        base_system_prompt = "You are a test agent."

    return SimpleAgent


@pytest.fixture
def agent_with_field_sets():
    """Create a ModelAgent with field sets."""

    class FieldSetAgent(ModelAgent):
        model = Place
        field_sets = {
            "public": ["name", "address"],
            "staff": ["name", "address", "phone", "notes"],
        }
        base_system_prompt = "Agent with field sets."

    return FieldSetAgent
