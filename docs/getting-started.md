# Getting started

## Installation

```console
uv add django-model-agent
```

Or with pip:

```console
pip install django-model-agent
```

### Django settings

If you plan to use `AgentMemory` (persistent memory tied to model instances),
add the app to `INSTALLED_APPS` and run migrations:

```python
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    # ...
    "django_model_agent",
]
```

```console
python manage.py migrate django_model_agent
```

If you only use `ModelAgent` and tools without memory, no Django app
registration or migrations are needed.

## Your first agent

Suppose you have a Django model:

```python
from django.db import models

class Restaurant(models.Model):
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255, blank=True)
    phone = models.CharField(max_length=50, blank=True)
    hours = models.TextField(blank=True)
    cuisine = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
```

Create an agent that can reason about it:

```python
from django_model_agent import ModelAgent

class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "phone", "hours", "cuisine"]

    _system_prompts = """
    You are an assistant that helps answer questions about restaurants.
    Use the provided model fields as your source of truth.
    """
```

That's it. The agent knows which fields to expose, generates a Pydantic schema
from the Django model, and combines your system prompts with a description of
the available fields and their current values.

All of that reaches the model as *instructions* rather than a system prompt, so
the field values are re-sent fresh each request instead of accumulating in the
conversation history. See the [migration guide](migration.md#system-prompts-are-now-instructions)
if you were relying on the old behaviour.

## Using the agent

```python
restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)
```

### Inspect without running

```python
# The auto-generated Pydantic schema
agent.schema
# >>> <class 'RestaurantAgentSchema'>

# Current field values
agent.get_current_values()
# >>> {'name': 'Zen Ramen', 'address': '42 Main St', ...}

# The combined system prompt
agent.get_system_prompts()

# Human-readable schema description
agent.get_schema_description()
```

## Configuring a model

Before an agent can call out to a provider, pydantic-ai needs two things: which
model to use, and credentials for it.

### Credentials

Providers read their keys from the environment. Set whichever matches the model
you plan to use:

```console
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."         # or GOOGLE_API_KEY
```

In Django, load these however you already handle secrets — `django-environ`, a
`.env` file, or your platform's config. They are read by pydantic-ai, not by
Django settings, so they must be present in the process environment.

### Choosing the model

Model names use pydantic-ai's `provider:model` form. Set it on the class:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "hours"]
    ai_model = "openai:gpt-4o"
```

or per instance:

```python
agent = RestaurantAgent(restaurant, ai_model="anthropic:claude-sonnet-4-20250514")
```

To drive it from settings rather than hardcoding:

```python
# settings.py
AGENT_MODEL = env("AGENT_MODEL", default="openai:gpt-4o")

# agents.py
from django.conf import settings

class RestaurantAgent(ModelAgent):
    model = Restaurant
    ai_model = settings.AGENT_MODEL
```

!!! note "`PYDANTIC_AI_MODEL` is not read by the library"

    You may see `PYDANTIC_AI_MODEL=...` in pydantic-ai's docs. That variable is
    read by their *example scripts*, not by pydantic-ai itself — setting it has
    no effect here. Use `ai_model`, wired to your own setting if you want
    environment control.

### Further reading

pydantic-ai owns model configuration; these are the pages worth having open:

- [Models overview](https://ai.pydantic.dev/models/) — every supported provider
  and its model-name format
- [OpenAI](https://ai.pydantic.dev/models/openai/) ·
  [Anthropic](https://ai.pydantic.dev/models/anthropic/) ·
  [Google](https://ai.pydantic.dev/models/google/) — per-provider setup,
  including custom base URLs and self-hosted or OpenAI-compatible endpoints
- [Model settings](https://ai.pydantic.dev/agents/#model-run-settings) —
  temperature, token limits, timeouts
- [Usage limits](https://ai.pydantic.dev/agents/#usage-limits) — capping
  requests and tokens so a runaway tool loop cannot surprise you

You can also pass a constructed model object instead of a string, which is how
you reach custom endpoints:

```python
from pydantic_ai.models.openai import OpenAIChatModel

agent = RestaurantAgent(restaurant, ai_model=OpenAIChatModel("gpt-4o"))
```

### Running prompts

django-model-agent integrates with pydantic-ai. Set the AI model on the class
or at init time:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "phone", "hours"]
    ai_model = "openai:gpt-4o"
    _system_prompts = "You help with restaurant information."
```

Then run prompts:

```python
restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)

# Async
result = await agent.run("What are the hours?")
print(result.output)

# Sync
result = agent.run_sync("What are the hours?")
print(result.output)
```

You can also override the model at init time:

```python
agent = RestaurantAgent(restaurant, ai_model="anthropic:claude-sonnet-4-20250514")
```

### Testing with pydantic-ai's TestModel

Use `build_agent()` and `override()` to test without making real API calls:

```python
from pydantic_ai.models.test import TestModel

agent = RestaurantAgent(restaurant)
pai = agent.build_agent()
agent._pydantic_agent = pai

with pai.override(model=TestModel()):
    result = agent.run_sync("What are the hours?")
    assert result.output is not None
```

## Decorator style

Instead of class attributes, you can use decorators to register system prompts,
instructions, and tools — closer to the pydantic-ai style:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours"]

    @ModelAgent.system_prompt
    def context_prompt(self) -> str:
        return "You help with restaurant information."

    @ModelAgent.instructions
    def dynamic_instructions(self) -> str:
        return f"Current restaurant: {self.instance.name}"

    @ModelAgent.tool
    def get_hours(self) -> str:
        """Get the restaurant hours."""
        return str(self.instance.hours)
```

Class-level attributes and decorators can be combined freely. The system prompt
will include all sources concatenated together.

## Field sets

Field sets let you expose different groups of fields depending on the
caller's role:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant

    _field_sets = {
        "public": ["name", "address", "phone"],
        "staff": ["name", "address", "phone", "notes", "internal_id"],
        "admin": None,  # None means all fields
    }

    _system_prompts = "You manage restaurant records."
```

Select a field set at init time:

```python
# Public user sees name, address, phone
agent = RestaurantAgent(restaurant, field_set="public")

# Staff sees additional fields
agent = RestaurantAgent(restaurant, field_set="staff")
```

## Overriding prompts at init time

System prompts and instructions can be overridden when creating an agent
instance:

```python
agent = RestaurantAgent(
    restaurant,
    system_prompt="You are a concise assistant. One sentence max.",
    instructions="Focus only on hours of operation.",
)
```

## Template-based instructions

For complex instructions that depend on model state, you can use Django
templates:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "hours"]
    _instructions_template = "agents/restaurant_instructions.html"
```

The template receives `instance` and `schema` in its context.
