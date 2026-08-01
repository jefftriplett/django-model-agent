# Capabilities

Capabilities are how django-model-agent extends a Pydantic AI agent. Each one
contributes some mix of instructions, tools, and lifecycle hooks, and they
compose — so you can take just the pieces you want, mix them with third-party
capabilities, and use them with a plain `pydantic_ai.Agent` if you never wanted
`ModelAgent` in the first place.

`ModelAgent` builds these for you. You only need this page if you want to add
capabilities to an agent, or skip `ModelAgent` entirely.

## Capabilities vs tools

If you have read the [Tools](tools.md) page, here is how the two relate.

A **tool** is one callable the model can invoke — a function with a schema.
A **capability** is a bundle of agent behaviour. It can *contain* tools, and it
can also do four things a tool cannot:

| | Tool | Capability |
|---|---|---|
| Be called by the model | yes | via the tools it carries |
| Contribute instructions | no | `get_instructions()` |
| Set model settings or pick the model | no | `get_model_settings()`, `get_model()` |
| Show or hide tools per request | per-tool `prepare` | `prepare_tools()` |
| Hook the run lifecycle | no | `before_run`, `after_run`, `wrap_run` |
| Keep per-run state | no | `for_run()` |

So they are layers, not alternatives. `GetPlaceInfoTool` stays a tool — it reads
some fields and returns them. `DjangoFSMCapability` has to be a capability,
because it both hides state-illegal tools *and* tells the agent which state the
instance is in; a tool has nowhere to put that second part.

Both are pydantic-ai concepts. `pydantic_ai.Agent` accepts `tools=`,
`toolsets=`, and `capabilities=`, and pydantic-ai's own extensions — `MCP`,
`Instrumentation`, `Thinking`, `ImageGeneration` — are all capabilities, because
none of them are expressible as a single function.

!!! note "Where the line blurs"

    `pydantic_ai.Tool` accepts a `prepare` hook — `(ctx, tool_def)` returning
    `None` to hide that tool — which overlaps with a capability's
    `prepare_tools()`. Filtering can legitimately live in either place. This
    library currently does it in `DjangoFSMCapability`.

## The four capabilities

| Capability | What it adds |
|------------|--------------|
| `DjangoModelCapability` | Field schema, current values, and the model's tools |
| `DjangoFSMCapability` | Current state, legal transitions, and state-aware tool filtering |
| `DjangoMemoryCapability` | Memory that persists across runs, keyed to the instance |
| `DjangoAuditCapability` | A record of what each run changed |

## Using them with a plain agent

Capabilities take a model **class**, never a model **instance**. The instance
arrives per-run through `deps`, so one agent serves every row:

```python
from pydantic_ai import Agent

from django_model_agent import ModelAgentContext
from django_model_agent.capabilities import (
    DjangoAuditCapability,
    DjangoFSMCapability,
    DjangoMemoryCapability,
    DjangoModelCapability,
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=ModelAgentContext,
    capabilities=[
        DjangoModelCapability(
            model_class=Place,
            fields=["name", "address", "phone", "state"],
            tools=[GetPlaceInfoTool, UpdateDescriptionTool],
            instructions="You are a helpful assistant.",
        ),
        DjangoFSMCapability(tools=[UpdateDescriptionTool]),
        DjangoMemoryCapability(max_history=50),
        DjangoAuditCapability(log_to="logger"),
    ],
)

place = Place.objects.get(pk=123)
result = await agent.run(
    "What is the address?",
    deps=ModelAgentContext(instance=place, agent=None),
)

# Same agent, a different row — no rebuild.
other = Place.objects.get(pk=456)
result = await agent.run(
    "Is this one open?",
    deps=ModelAgentContext(instance=other, agent=None),
)
```

Building one agent per model instance would work, but it is wasted effort:
nothing in the agent depends on which row you are looking at.

## Adding capabilities to a ModelAgent

Override `get_extra_capabilities()`:

```python
from django_model_agent import ModelAgent
from django_model_agent.capabilities import DjangoAuditCapability


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "address"]
    tools = [GetPlaceInfoTool]

    def get_extra_capabilities(self):
        return [DjangoAuditCapability(log_to="logger")]
```

`ModelAgent` always adds `DjangoModelCapability`. It adds `DjangoFSMCapability`
when one of your tools actually restricts by `allowed_states`, and
`DjangoMemoryCapability` when the agent uses `AgentMemoryMixin`.

## DjangoModelCapability

Describes the model to the agent and exposes its tools.

```python
DjangoModelCapability(
    model_class=Place,
    fields=["name", "address"],   # None exposes every field
    exclude=["internal_notes"],
    tools=[GetPlaceInfoTool],
    instructions="You are a helpful assistant.",
)
```

Field values are read fresh on every request, so a value a tool changes
mid-conversation is visible on the next turn.

### Why instructions and not a system prompt

The current field values go out as *instructions*, not as a system prompt.

A system prompt is written into the message history and stays there. Because
this capability injects live field values, a system prompt would leave turn
one's values sitting in the context on turn five, contradicting the values that
are actually current. Instructions are re-sent fresh on each request and are
kept out of history, so there is only ever one set of values in play.

This is why `ModelAgent._system_prompts` is delivered as instructions too.

## DjangoFSMCapability

Tells the agent which state the instance is in, and hides tools that cannot run
in that state.

```python
DjangoFSMCapability(
    state_field="state",
    tools=[PublishTool, CloseTool],
    transitions={
        "draft": ["public"],
        "public": ["featured", "closed"],
        "featured": ["public"],
        "closed": [],
    },
)
```

Tools declare where they are legal:

```python
class PublishTool(UpdateTool):
    name = "publish"
    description = "Publish this place"
    allowed_states = ["draft"]
```

Hiding a tool is an optimisation, not the enforcement point. `ModelTool` still
checks `allowed_states` when it runs, so `allowed_states` behaves the same
whether or not you add this capability. What the capability buys you is that
the model never sees a tool it cannot use — so it does not spend tokens calling
one, and a refusal never lands in the conversation.

Tools with no `allowed_states` are never filtered, and a model with no state
field is left alone.

## DjangoMemoryCapability

Loads memory before a run and saves it after, keyed to the instance.

```python
DjangoMemoryCapability(
    max_history=50,        # turns to keep before trimming oldest
    include_history=True,  # feed past turns back as instructions
)
```

Backed by the [`AgentMemory`](memory.md) model, so it needs
`django_model_agent` in `INSTALLED_APPS` and its migration applied.

Instances that have never been saved are skipped rather than raising — an
unsaved model has no primary key to key memory against.

## DjangoAuditCapability

Snapshots the instance before a run, diffs it after, and reports what changed.

```python
def record_change(record):
    AuditLog.objects.create(
        obj_pk=record.instance_pk,
        summary=record.summary(),
        tools=record.tool_calls,
    )

DjangoAuditCapability(log_to="callback", callback=record_change)
```

`log_to` accepts:

| Value | Effect |
|-------|--------|
| `"logger"` | Writes a summary to the `django_model_agent.capabilities` logger |
| `"callback"` | Hands an `AuditRecord` to your `callback` |
| `"none"` | Collects the record without reporting it |

An `AuditRecord` carries `instance_pk`, `model_class`, `prompt`,
`field_changes`, and `tool_calls`, plus a `changed` flag and a `summary()`.

Pass `track_fields=[...]` to watch specific fields instead of every editable one.

!!! note "Use the callback to collect records"

    Every run gets its own copy of the capability, so the `record` attribute on
    the instance you constructed stays `None`. The callback is how records get
    out — and how you persist them to your own audit table.

## Writing your own

Capabilities are plain Pydantic AI capabilities. Subclass `AbstractCapability`
and implement the hooks you need:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from django_model_agent import ModelAgentContext


class TenantCapability(AbstractCapability[ModelAgentContext]):
    """Remind the agent which tenant it is acting for."""

    def get_instructions(self):
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            return f"You are acting for tenant {ctx.deps.instance.tenant_id}."

        return _instructions
```

Read state from `ctx.deps` rather than capturing it in `__init__`, so your
capability stays usable across instances.

If a hook needs the database, do that work in `before_run` or `after_run`
wrapped in `asgiref.sync.sync_to_async` — those hooks are async and Django's
ORM is not. Keep `get_instructions()` callables free of queries; have them read
what an earlier hook already loaded.

See the [Pydantic AI capabilities docs](https://ai.pydantic.dev/capabilities/overview/)
for the full set of hooks.
