"""
Model registration for the django_model_agent app.

``AgentMemory`` is defined in ``memory.py`` so it can be imported directly
without pulling in the rest of the package. Django only discovers models
through ``<app>/models.py``, though, so it is re-exported here -- without this
the model is never registered, no table is created, and the documented
``manage.py migrate django_model_agent`` has nothing to apply.

Importing from ``django_model_agent.memory`` keeps working unchanged.
"""

from .memory import AgentMemory, AgentMemoryManager

__all__ = ["AgentMemory", "AgentMemoryManager"]
