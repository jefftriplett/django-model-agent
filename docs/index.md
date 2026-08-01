# django-model-agent

A Django Ninja-style abstraction for binding Django models to Pydantic AI Agents.

```python
from django_model_agent import ModelAgent

class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours"]

    _system_prompts = "You help with restaurant information."

    @ModelAgent.tool
    def get_hours(self) -> str:
        """Get the restaurant hours."""
        return str(self.instance.hours)

restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)
result = await agent.run("Are we open on Christmas Day?")
```

Define which model fields the agent sees, register system prompts and tools
declaratively, and run prompts through pydantic-ai — all from a single class.

## Why this exists

Building AI agents that operate on Django model instances requires a lot of
glue: extracting field values, building schemas, wiring up tools, managing
system prompts. django-model-agent provides a declarative layer that handles
all of that, similar to how Django Ninja's `ModelSchema` handles serialization.

<div class="grid cards" markdown>

-   __Declarative binding__

    Define which Django model fields your agent can access with a simple list.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   __Tools__

    Register tools as class attributes or decorated methods with automatic
    state checking and diff-aware updates.

    [:octicons-arrow-right-24: Tools](tools.md)

-   __Persistent memory__

    Attach memory to any Django model instance via the contenttypes framework.

    [:octicons-arrow-right-24: Memory](memory.md)

-   __Pydantic AI integration__

    `build_agent()` produces a real `pydantic_ai.Agent` with deps, tools,
    and prompts wired up.

    [:octicons-arrow-right-24: API reference](reference.md)

</div>

## Install

```console
uv add django-model-agent
```

Or with pip:

```console
pip install django-model-agent
```

Add `django_model_agent` to your `INSTALLED_APPS` if you want to use the
`AgentMemory` model:

```python
INSTALLED_APPS = [
    # ...
    "django_model_agent",
]
```

## A tour in one page

```python
from django_model_agent import ModelAgent

# Declare the agent
class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "address", "phone", "state"]
    _system_prompts = "You manage place records."

    @ModelAgent.system_prompt
    def state_context(self) -> str:
        return f"Current place: {self.instance.name} ({self.instance.state})"

    @ModelAgent.tool
    def get_phone(self) -> str:
        """Get the phone number for this place."""
        return self.instance.phone or "No phone on file"

# Use it
place = Place.objects.get(pk=42)
agent = PlaceAgent(place, ai_model="openai:gpt-4o")

# Inspect without running
agent.schema              # Pydantic model with name, address, phone, state
agent.get_system_prompts() # Combined system prompt text
agent.get_tools()          # List of tool callables

# Run a prompt
result = await agent.run("What is the phone number?")
print(result.output)
```

## llms.txt

This documentation is available in the [llms.txt](https://llmstxt.org/)
format, a Markdown convention suited to LLMs and AI coding assistants.

Two files are published:

- [`llms.txt`](https://jefftriplett.github.io/django-model-agent/llms.txt): a
  short description of the project plus links to each section.
- [`llms-full.txt`](https://jefftriplett.github.io/django-model-agent/llms-full.txt):
  the same index with the content of every page inlined.

## Where to next

- [Getting started](getting-started.md) — installation, first agent, running prompts
- [Tools](tools.md) — ModelTool, ReadOnlyTool, UpdateTool, DiffAwareUpdateTool
- [Memory](memory.md) — AgentMemory model, AgentMemoryMixin
- [API reference](reference.md) — every class, method, and attribute
- [Contributing](contributing.md) — setup, testing, conventions

## License

PolyForm Noncommercial License 1.0.0
