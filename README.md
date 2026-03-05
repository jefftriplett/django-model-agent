# Django Model Agent

> **Note:** This project is experimental and the API will change.

A Django Ninja-style abstraction for binding Django models to Pydantic AI Agents.

This library provides a declarative way to create AI agents that understand and operate on Django model instances, similar to how Django Ninja's ModelSchema provides a declarative way to serialize models.

## Installation

This package is not yet published on PyPI. Install directly from GitHub:

```bash
uv add git+https://github.com/jefftriplett/django-model-agent
```

Or with pip:

```bash
pip install git+https://github.com/jefftriplett/django-model-agent
```

Once published on PyPI:

```bash
uv add django-model-agent
```

Or with pip:

```bash
pip install django-model-agent
```

## Quick Start

### Class Attribute Style

```python
from django_model_agent import ModelAgent, ModelTool


class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours", "neighborhood"]

    _system_prompts = """
    You are an assistant that helps reason about restaurant information.
    Use the provided model fields as your source of truth.
    """

    tools = [UpdateHoursTool, FlagForReviewTool]
```

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

### Usage

```python
restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)
result = await agent.run("Are you open on Christmas Day?")

# Override prompts at init time
agent = RestaurantAgent(
    restaurant,
    system_prompt="You are a concise assistant.",
    instructions="Focus only on hours of operation.",
)
```

## Features

- **Declarative model binding**: Define which Django model fields your agent can access
- **Field sets**: Create named groups of fields for role-based exposure
- **Multiple prompt sources**: Combine class-level prompts with decorated methods
- **Tool registration**: Register tools via class attributes or decorators
- **Template support**: Use Django templates for dynamic instructions
- **Automatic schema generation**: Pydantic schemas are generated from Django models

## Class Attributes

| Attribute | Description |
|-----------|-------------|
| `model` | The Django model class this agent operates on |
| `fields` | List of field names to expose to the agent (None = all fields) |
| `exclude` | List of field names to exclude from the schema |
| `_system_prompts` | System prompt string or list of strings for the agent |
| `_instructions` | Instructions string or list of strings for the agent |
| `_instructions_template` | Path to a Django template for instructions |
| `tools` | List of tool classes available to the agent |
| `_field_sets` | Named groups of fields for role-based exposure |

## Init Parameters

| Parameter | Description |
|-----------|-------------|
| `instance` | The Django model instance to operate on |
| `system_prompt` | Override or extend the class-level system prompts |
| `instructions` | Override or extend the class-level instructions |
| `field_set` | Name of a field set to use for schema generation |

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

## License

PolyForm Noncommercial License 1.0.0
