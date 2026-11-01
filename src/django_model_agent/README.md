# django_model_agent

A Django Ninja-style way to bind Django models to Pydantic AI agents. Full
documentation: https://django-model-agent.readthedocs.io/en/latest/

```
Django Model = source of truth
ModelAgent   = declarative wrapper
Pydantic AI  = execution engine
```

```python
from django_model_agent import ModelAgent


class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "address", "hours"]
    ai_model = "openai:gpt-4o"
    _system_prompts = "Use the provided model fields as your source of truth."
    tools = [UpdateHoursTool]


result = await RestaurantAgent(restaurant).run("Are we open on Christmas Day?")
```

## Modules

| Module | Contents |
|--------|----------|
| `base.py` | `ModelAgent`, which composes capabilities into a `pydantic_ai.Agent` and runs it with a `ModelAgentContext` as deps; the `@ModelAgent.system_prompt`, `@ModelAgent.instructions`, and `@ModelAgent.tool` decorators |
| `capabilities.py` | `DjangoModelCapability` (field schema, live values, tools), `DjangoFSMCapability` (state-aware tool filtering), `DjangoMemoryCapability` (per-instance conversation memory), `DjangoAuditCapability` (field changes, tool calls, usage); helpers that wrap `ModelTool` classes as pydantic-ai tools |
| `tools.py` | `ModelTool`, `ReadOnlyTool`, `UpdateTool`, `ToolResult`; `DiffAwareUpdateTool` is deprecated in favour of `requires_confirmation` |
| `memory.py` | The `AgentMemory` model and its manager; `AgentMemoryMixin` is deprecated in favour of `DjangoMemoryCapability` |
| `models.py` | Re-exports `AgentMemory` so Django registers it |
| `examples.py` | Example agents and tools for a place-like model |

## How a run works

1. `ModelAgent.build_agent()` creates a `pydantic_ai.Agent` with
   `deps_type=ModelAgentContext` and the capabilities from
   `_build_capabilities()` plus `get_extra_capabilities()`.
2. Capabilities hold the model *class*, never an instance. The instance arrives
   per run through `ctx.deps.instance`, so one built agent can serve any row.
3. Field schema and current values are sent as instructions, re-evaluated on
   every model request and kept out of message history.
4. Sync tools run in a worker thread and close any database connection they
   open; async tools run on the event loop and must use Django's async ORM.
