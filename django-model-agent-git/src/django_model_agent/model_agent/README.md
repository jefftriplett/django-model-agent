# ModelAgent - Django Ninja-style Abstraction for Pydantic AI

This experimental module provides a declarative way to bind Django models to Pydantic AI Agents,
inspired by Django Ninja's `ModelSchema` pattern.

## Concept

Just as Django Ninja's `ModelSchema` provides a declarative way to create Pydantic schemas
from Django models for API serialization, `ModelAgent` provides a declarative way to create
AI agents that understand and operate on Django model instances.

```
Django Model = source of truth
ModelAgent   = declarative wrapper
Pydantic AI  = execution engine
```

## Quick Start

```python
from experimental.model_agent import ModelAgent, ModelTool


class RestaurantAgent(ModelAgent):
    model = Restaurant

    fields = ["name", "address", "hours", "neighborhood"]

    base_prompt = """
    You are an assistant that helps reason about restaurant information.
    Use the provided model fields as your source of truth.
    """

    instructions_template = "agents/restaurant_instructions.jinja"

    tools = [UpdateHoursTool, FlagForReviewTool]


# Usage
restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)
result = await agent.run("Are we open on Christmas Day?")
```

## Core Components

### ModelAgent (`base.py`)

The base class that:
1. Binds to a Django model
2. Generates a Pydantic schema from model fields
3. Creates a Pydantic AI Agent
4. Handles loading instances and persisting changes

Key attributes:
- `model`: The Django model class
- `fields`: List of fields to expose (None = all)
- `exclude`: Fields to exclude from schema
- `base_prompt`: Base prompt for the agent (combined with `@system_prompt` decorators)
- `instructions_template`: Path to Django template for dynamic instructions
- `tools`: List of tool classes available to the agent
- `field_sets`: Named groups of fields for role-based access

### ModelTool (`tools.py`)

Base class for creating tools that operate on model instances:

```python
class UpdateHoursTool(ModelTool):
    name = "update_hours"
    description = "Update restaurant hours"

    async def execute(self, *, hours: str) -> ToolResult:
        self.instance.hours = hours
        self.instance.save()
        return ToolResult(success=True, message="Hours updated")
```

Specialized tool types:
- `ReadOnlyTool`: For read-only operations
- `UpdateTool`: For simple updates with change tracking
- `DiffAwareUpdateTool`: For proposing changes that require approval

### AgentMemory (`memory.py`)

Model-backed memory storage using Django's contenttypes framework:

```python
class AgentMemory(models.Model):
    content_type = ForeignKey(ContentType)
    object_id = PositiveIntegerField()
    data = JSONField()
```

Add memory to your agent with the mixin:

```python
class RestaurantAgent(AgentMemoryMixin, ModelAgent):
    model = Restaurant
    ...


agent = RestaurantAgent(restaurant)
agent.memory.set("last_topic", "hours")
agent.save_memory()
```

## Decorators (pydantic-ai style)

ModelAgent supports pydantic-ai style decorators for registering prompts, instructions, and tools:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours"]
    base_prompt = "Base system prompt."  # Optional class attribute

    @ModelAgent.system_prompt
    def dynamic_context(self) -> str:
        """Added to the system prompt at runtime."""
        return f"Current restaurant: {self.instance.name}"

    @ModelAgent.instructions
    def state_instructions(self) -> str:
        """Dynamic instructions that change per-run."""
        return f"Current state: {self.instance.state}"

    @ModelAgent.tool
    def get_hours(self) -> str:
        """Get the operating hours for this restaurant."""
        return str(self.instance.hours)

    @ModelAgent.tool
    def update_name(self, new_name: str) -> str:
        """Update the restaurant name."""
        self.instance.name = new_name
        self.instance.save()
        return f"Name updated to: {new_name}"
```

### Decorator Reference

| Decorator | pydantic-ai equivalent | Purpose |
|-----------|----------------------|---------|
| `@ModelAgent.system_prompt` | `@agent.system_prompt` | Add to system prompt |
| `@ModelAgent.instructions` | `@agent.instructions` | Dynamic per-run instructions |
| `@ModelAgent.tool` | `@agent.tool` | Register a tool |

Multiple decorated methods of the same type are combined:

```python
class Agent(ModelAgent):
    base_prompt = "Base prompt."

    @ModelAgent.system_prompt
    def context_1(self) -> str:
        return "Context part 1."

    @ModelAgent.system_prompt
    def context_2(self) -> str:
        return "Context part 2."


# Result: "Base prompt.\n\nContext part 1.\n\nContext part 2."
```

## Field Sets (Role-Based Access)

Expose different fields based on user role:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant

    field_sets = {
        "public": ["name", "hours", "address"],
        "staff": ["name", "hours", "address", "internal_notes"],
        "admin": None,  # All fields
    }


# Use a specific field set
agent = RestaurantAgent(restaurant, field_set="staff")
```

## Multi-Agent Workflows

### Diff-Aware Updates

One agent proposes, another approves:

```python
# Proposer agent
class DataEntryAgent(ModelAgent):
    tools = [ProposeChangeTool]


# Reviewer agent
class ReviewerAgent(ModelAgent):
    tools = [ApproveChangeTool, RejectChangeTool]
```

### Tool with FSM State Checking

Tools can restrict themselves to certain model states:

```python
class PublishTool(ModelTool):
    name = "publish"
    allowed_states = ["draft"]  # Only works when state is "draft"

    async def execute(self, **kwargs):
        self.instance.publish()
        self.instance.save()
        return ToolResult(success=True, message="Published")
```

## Integration with Django Admin

You could wire ModelAgent into Django Admin actions:

```python
@admin.action(description="Run agent task")
def run_agent_task(modeladmin, request, queryset):
    for instance in queryset:
        agent = RestaurantAgent(instance)
        # ... run agent task
```

## TODO / Future Directions

- [ ] Full integration with pydantic-ai when API stabilizes
- [ ] Streaming response support
- [ ] Tool result caching
- [ ] Batch operations across multiple instances
- [ ] Admin integration helpers
- [ ] Django management commands for agent tasks
- [ ] Webhook/callback support for async workflows
