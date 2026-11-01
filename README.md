# Django Model Agent

> **Note:** This project is experimental and the API will change.

A Django Ninja-style abstraction for binding Django models to Pydantic AI Agents.

**[Documentation](https://django-model-agent.readthedocs.io/en/latest/)** · [Getting started](https://django-model-agent.readthedocs.io/en/latest/getting-started/) · [Cookbook](https://django-model-agent.readthedocs.io/en/latest/cookbook/) · [Examples](https://django-model-agent.readthedocs.io/en/latest/examples/)

This library provides a declarative way to create AI agents that understand and operate on Django model instances, similar to how Django Ninja's ModelSchema provides a declarative way to serialize models.

## Installation

```bash
uv add django-model-agent
```

Or with pip:

```bash
pip install django-model-agent
```

Requires Python 3.13+, Django 5.2+, and pydantic-ai 2.22+.

## Quick Start

Declare which fields the agent can see, give it instructions, and run it against
an instance:

```python
from django_model_agent import ModelAgent


class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours", "neighborhood"]
    ai_model = "openai:gpt-4o"

    _system_prompts = """
    You are an assistant that helps reason about restaurant information.
    Use the provided model fields as your source of truth.
    """

    tools = [UpdateHoursTool, FlagForReviewTool]


restaurant = await Restaurant.objects.aget(pk=123)
result = await RestaurantAgent(restaurant).run("Are we open on Christmas Day?")
print(result.output)
```

The field schema and the instance's current values reach the model as
instructions, re-read on every request, so they're never stale. `run_sync()`,
`run_stream()`, and `run_stream_events()` are there too.

### Decorator Style (Pydantic-AI inspired)

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

### Overrides

Most class attributes can be overridden per agent:

```python
agent = RestaurantAgent(
    restaurant,
    instructions="Focus only on hours of operation.",
    ai_model="anthropic:claude-sonnet-4-20250514",
    output_type=HoursAnswer,
)
```

When no model is set anywhere, the `PYDANTIC_AI_MODEL` environment variable is
used.

## Features

- **Declarative model binding**: choose which Django model fields the agent can see
- **Field sets**: named groups of fields for role-based exposure
- **Tools**: `ModelTool`, `ReadOnlyTool`, and `UpdateTool` classes, or decorated
  methods; sync or async
- **Structured output**: set `output_type` to get a validated Pydantic model back
- **Human approval**: `requires_confirmation = True` suspends the run before a tool executes
- **Workflow state**: `allowed_states` hides tools that can't run in the
  instance's current state
- **Memory**: `DjangoMemoryCapability` keeps conversations per instance in the database
- **Audit trail**: `DjangoAuditCapability` reports field changes, tool calls, and token usage
- **Usage limits**: cap requests, tool calls, and tokens per run, agent, or project
- **Composable**: everything is a pydantic-ai capability, usable with a plain
  `pydantic_ai.Agent` too

## Init Parameters

| Parameter | Description |
|-----------|-------------|
| `instance` | The Django model instance to operate on |
| `system_prompt` | Override the class-level system prompts |
| `instructions` | Override the class-level instructions |
| `field_set` | Name of a field set to use for schema generation |
| `ai_model` | The pydantic-ai model to use, e.g. `"openai:gpt-4o"` |
| `output_type` | Structured output type for this agent |
| `usage_limits` | `UsageLimits` applied to every run |

## Class Attributes

| Attribute | Description |
|-----------|-------------|
| `model` | The Django model class this agent operates on |
| `fields` | List of field names to expose to the agent (None = all fields) |
| `exclude` | List of field names to exclude from the schema |
| `ai_model` | Default pydantic-ai model |
| `output_type` | Default structured output type (None = plain text) |
| `usage_limits` | Default `UsageLimits` |
| `_system_prompts` | System prompt string or list of strings for the agent |
| `_instructions` | Instructions string or list of strings for the agent |
| `_instructions_template` | Path to a Django template for instructions |
| `tools` | List of tool classes available to the agent |
| `_field_sets` | Named groups of fields for role-based exposure |

Override `get_extra_capabilities()` to add capabilities such as memory, auditing,
or any pydantic-ai capability.

## Decorators

| Decorator | Description |
|-----------|-------------|
| `@ModelAgent.system_prompt` | Register a method as a system prompt provider |
| `@ModelAgent.instructions` | Register a method as an instructions provider |
| `@ModelAgent.tool` | Register a method as a tool available to the agent |

## Creating Tools

```python
from django_model_agent.tools import UpdateTool, ToolResult


class UpdateHoursTool(UpdateTool):
    name = "update_hours"
    description = "Update the operating hours"

    def update(self, *, new_hours: str) -> None:
        self.instance.hours = new_hours
        # save() is called automatically by UpdateTool
```

Or for read-only tools:

```python
from django_model_agent.tools import ReadOnlyTool


class GetHoursTool(ReadOnlyTool):
    name = "get_hours"
    description = "Get the current operating hours"

    def read(self, **kwargs) -> dict:
        return {"hours": self.instance.hours}
```

## Local Development

Python `3.13+` is currently required, including free-threaded (`t`) builds.

```bash
uv sync --group dev
just test
just lint
```

Useful commands:

- `just test tests/test_model_agent.py`
- `just lint`
- `uv build`

## License

PolyForm Noncommercial License 1.0.0
