# API reference

## ModelAgent

The base class for creating AI agents bound to Django models.

```python
from django_model_agent import ModelAgent
```

### Class attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `type[models.Model]` | — | The Django model class this agent operates on |
| `fields` | `list[str] \| None` | `None` | Field names to expose (`None` = all fields) |
| `exclude` | `list[str]` | `[]` | Field names to exclude from the schema |
| `_system_prompts` | `str \| list[str]` | `""` | System prompt string or list of strings |
| `_instructions` | `str \| list[str]` | `""` | Instructions string or list of strings |
| `_instructions_template` | `str \| None` | `None` | Path to a Django template for instructions |
| `tools` | `Sequence[Any]` | `[]` | Tool classes available to the agent |
| `_field_sets` | `dict[str, list[str]]` | `{}` | Named groups of fields for role-based exposure |
| `ai_model` | `str \| None` | `None` | The pydantic-ai model name (e.g. `'openai:gpt-4o'`); falls back to `PYDANTIC_AI_MODEL` |
| `output_type` | `Any` | `None` | Structured output type; `None` gives plain string output |
| `usage_limits` | `UsageLimits \| None` | `None` | Default limits; falls back to the `DJANGO_MODEL_AGENT_USAGE_LIMITS` setting |

### `__init__`

```python
ModelAgent(
    instance: models.Model,
    *,
    system_prompt: str | list[str] | None = None,
    instructions: str | list[str] | None = None,
    field_set: str | None = None,
    ai_model: str | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `instance` | The Django model instance to operate on |
| `system_prompt` | Override or extend the class-level system prompts |
| `instructions` | Override or extend the class-level instructions |
| `field_set` | Name of a field set to use for schema generation |
| `ai_model` | Override the pydantic-ai model to use |
| `output_type` | Override the structured output type |
| `usage_limits` | Override the default usage limits |

### Properties

`schema`
:   Lazily built and cached Pydantic `BaseModel` generated from the Django model
    fields. Field types are mapped from Django field types to Python types.

`context`
:   A `ModelAgentContext` instance providing access to the model instance and agent.

### Methods

`get_system_prompts() -> str`
:   Get the combined system prompt. Concatenates class-level `_system_prompts`
    with all `@ModelAgent.system_prompt` decorated methods.

`get_instructions() -> str | None`
:   Get the combined instructions. Concatenates class-level `_instructions`,
    rendered `_instructions_template`, and all `@ModelAgent.instructions`
    decorated methods. Returns `None` if no instructions are defined.

`get_tools() -> list[Callable]`
:   Get all tools. Combines class-level `tools` list with
    `@ModelAgent.tool` decorated methods.

`get_schema_description() -> str`
:   Human-readable description of the schema fields and their types.

`get_current_values() -> dict[str, Any]`
:   Current values of all schema fields from the instance.

`build_agent() -> pydantic_ai.Agent`
:   Build a `pydantic_ai.Agent` with `deps_type=ModelAgentContext`, composed
    from [capabilities](capabilities.md). The schema description and current
    values are delivered as instructions rather than a system prompt, so stale
    values never accumulate in message history.

`_get_ai_model() -> str | Model | None`
:   Resolve the model: `__init__` argument, then class `ai_model`, then the
    `PYDANTIC_AI_MODEL` environment variable.

`_get_output_type() -> Any`
:   Resolve the structured output type: `__init__` argument, then class
    `output_type`. `None` leaves pydantic-ai's plain string output in place.

`get_extra_capabilities() -> list[AbstractCapability]`
:   Additional capabilities to compose into the agent. Override to add
    capabilities such as `DjangoAuditCapability`. Returns an empty list by
    default.

`run(prompt: str, **kwargs) -> AgentRunResult` *(async)*
:   Run the agent with a prompt through pydantic-ai. Lazily calls
    `build_agent()` on first use. Extra kwargs are forwarded to
    `pydantic_ai.Agent.run()`.

`run_sync(prompt: str, **kwargs) -> AgentRunResult`
:   Synchronous version of `run()`. Extra kwargs are forwarded to
    `pydantic_ai.Agent.run_sync()`.

### Decorators

`@ModelAgent.system_prompt`
:   Register a method as a system prompt provider. The method should return a
    string. Multiple methods can be decorated.

`@ModelAgent.instructions`
:   Register a method as an instructions provider. Instructions are dynamic
    guidance that can change per-run.

`@ModelAgent.tool`
:   Register a method as a tool. The method's docstring becomes the tool
    description, and the method signature defines the tool's parameters.

### Django field type mapping

| Django field | Python type |
|--------------|-------------|
| `AutoField`, `BigAutoField`, `SmallAutoField` | `int` |
| `IntegerField`, `SmallIntegerField`, `BigIntegerField` | `int` |
| `PositiveIntegerField`, `PositiveSmallIntegerField`, `PositiveBigIntegerField` | `int` |
| `FloatField` | `float` |
| `DecimalField` | `Decimal` |
| `CharField`, `TextField`, `EmailField`, `URLField`, `SlugField` | `str` |
| `UUIDField`, `FilePathField`, `FileField`, `ImageField` | `str` |
| `GenericIPAddressField`, `IPAddressField` | `str` |
| `BooleanField` | `bool` |
| `NullBooleanField` | `Optional[bool]` |
| `DateField`, `DateTimeField`, `TimeField`, `DurationField` | `str` |
| `BinaryField` | `bytes` |
| `JSONField` | `dict` |
| `ForeignKey` | `int` |

Nullable fields are wrapped in `Optional[...]`.

## ModelAgentContext

Context object passed to tools, providing access to the model instance.

```python
from django_model_agent import ModelAgentContext
```

| Attribute | Description |
|-----------|-------------|
| `instance` | The Django model instance |
| `agent` | The parent `ModelAgent` instance |

`refresh_instance()`
:   Reload the instance from the database.

## ToolResult

```python
from django_model_agent.tools import ToolResult
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `success` | `bool` | — | Whether the tool executed successfully |
| `message` | `str` | — | Human-readable result message |
| `data` | `dict \| None` | `None` | Optional structured data |
| `changes` | `dict` | `{}` | Field changes made (for audit/diff) |

## ModelTool

```python
from django_model_agent.tools import ModelTool
```

Abstract base class for tools that operate on Django model instances.

| Class attribute | Type | Default | Description |
|----------------|------|---------|-------------|
| `name` | `str` | — | Unique tool identifier |
| `description` | `str` | — | Human-readable description for the AI |
| `requires_confirmation` | `bool` | `False` | Advisory only; not enforced by the library |
| `allowed_states` | `list[str] \| None` | `None` | FSM states where allowed |

**Abstract method:** `execute(**kwargs) -> ToolResult`

## ReadOnlyTool

```python
from django_model_agent.tools import ReadOnlyTool
```

Base class for tools that only read data.

**Abstract method:** `read(**kwargs) -> dict[str, Any]`

## UpdateTool

```python
from django_model_agent.tools import UpdateTool
```

Base class for tools that update model fields. Captures state before/after,
computes diff, and calls `save()` automatically.

`requires_confirmation` defaults to `True`, but is advisory only — it does not
gate the save. Pass `preview=True` to inspect changes without writing.

**Abstract method:** `update(**kwargs) -> None`

## DiffAwareUpdateTool

```python
from django_model_agent.tools import DiffAwareUpdateTool
```

Tool that proposes changes instead of applying them directly.

`propose_change(field_name, new_value, reason="") -> ProposedChange`
:   Propose a change to a field.

`get_pending_changes() -> list[ProposedChange]`
:   Get changes that haven't been approved or rejected.

`apply_approved_changes() -> int`
:   Apply all approved changes and save. Returns count of changes applied.

`get_diff_summary() -> str`
:   Human-readable summary of proposed changes.

## ProposedChange

```python
from django_model_agent.tools import ProposedChange
```

| Attribute | Description |
|-----------|-------------|
| `instance` | The model instance |
| `field_name` | Name of the field being changed |
| `old_value` | Current value |
| `new_value` | Proposed new value |
| `reason` | Why this change is proposed |
| `approved` | `True`, `False`, or `None` |

`approve()` / `reject()` / `apply()`

## AgentMemory

```python
from django_model_agent.memory import AgentMemory
```

Django model that stores agent memory/state tied to any model instance via
`GenericForeignKey`. See [Memory](memory.md) for full documentation.

## Capabilities

See [Capabilities](capabilities.md) for usage and examples.

`DjangoModelCapability(*, model_class, fields=None, exclude=None, tools=(), tool_funcs=(), instructions="")`
:   Contributes the field schema, current values, and the model's tools. Takes
    a model class, never an instance -- the instance arrives per-run via
    `ctx.deps.instance`.

`DjangoFSMCapability(*, state_field="state", tools=(), tool_states=None, transitions=None)`
:   Contributes the current state and legal transitions, and hides tools whose
    `allowed_states` exclude the current state. Filtering is an optimisation;
    `ModelTool.check_allowed()` remains the enforcement point.

`DjangoMemoryCapability(*, max_history=100, include_history=True)`
:   Loads `AgentMemory` before a run and saves it after. Skips instances with
    no primary key.

`DjangoAuditCapability(*, log_to="logger", callback=None, track_fields=None)`
:   Snapshots the instance before a run and diffs it after, producing an
    `AuditRecord`. `log_to` accepts `"logger"`, `"callback"`, or `"none"`.

## AuditRecord

Dataclass describing what one run did.

| Attribute | Type | Description |
|-----------|------|-------------|
| `instance_pk` | `Any` | Primary key of the audited instance |
| `model_class` | `str` | Name of the model class |
| `prompt` | `str` | The prompt that drove the run |
| `field_changes` | `dict[str, dict[str, Any]]` | `{field: {before, after}}` |
| `tool_calls` | `list[dict[str, Any]]` | `{name, args}` per call |
| `usage` | `dict[str, int]` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `requests`, `tool_calls` |

`total_tokens -> int`
:   Input plus output tokens.

`changed -> bool`
:   Whether any field changed.

`summary() -> str`
:   Human-readable one-line summary of the changes.

## AgentMemoryMixin

```python
from django_model_agent.memory import AgentMemoryMixin
```

Mixin for `ModelAgent` to add persistent memory support. See
[Memory](memory.md) for full documentation.
