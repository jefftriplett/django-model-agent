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

### When instructions are re-evaluated

Not everything that ends up in the instructions is refreshed at the same time,
and the difference bites when an instance changes underneath a long-lived agent.

| Source | Evaluated |
|---|---|
| `DjangoModelCapability` schema and current values | **every request** |
| Any capability's `get_instructions()` callable | **every request** |
| `_system_prompts` / `_instructions` class attributes | once, when the agent is built |
| `@ModelAgent.system_prompt` / `@ModelAgent.instructions` methods | once, when the agent is built |
| `_instructions_template` | once, when the agent is built |

`ModelAgent` collects the class-level prompts, decorated methods, and rendered
template into text at `build_agent()` time and hands that text to the capability
as static instructions. The capability then adds the live values on top.

`build_agent()` runs once per `ModelAgent` — the result is cached on
`_pydantic_agent` at first use — so in practice everything in the bottom half of
that table is evaluated once per agent object.

!!! warning "A decorated `@ModelAgent.instructions` method does not re-run"

    The name suggests otherwise, and in pydantic-ai `instructions` *are* per
    request. Here the method is called once and its output frozen:

    ```python
    class PlaceAgent(ModelAgent):
        model = Place

        @ModelAgent.instructions
        def state_hint(self) -> str:
            return f"The state is {self.instance.state}."   # evaluated ONCE


    agent = PlaceAgent(place)
    await agent.run("...")            # "The state is draft."
    place.state = "public"
    await agent.run("...")            # still "The state is draft."
    ```

    The field values that `DjangoModelCapability` injects *would* be correct in
    that second run. Only the decorated text is stale — which is worse than if
    everything were stale, because the two now disagree.

Two ways out. Build a new agent when the instance changes, which is cheap and is
what most code does anyway:

```python
for place in Place.objects.all():
    await PlaceAgent(place).run("...")      # fresh agent, fresh text
```

Or, if the agent must be long-lived, put the changing text in a capability so it
is evaluated per request:

```python
class StateHintCapability(AbstractCapability[ModelAgentContext]):
    def get_instructions(self):
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            return f"The state is {ctx.deps.instance.state}."   # per request
        return _instructions


class PlaceAgent(ModelAgent):
    model = Place

    def get_extra_capabilities(self):
        return [StateHintCapability()]
```

### Returning a string vs a callable

Inside a capability the same distinction applies, and it is easy to trip over:

```python
def get_instructions(self):
    place = self.instance                       # WRONG on two counts
    return f"Reviewing {place.name}."           # evaluated once, and there is
                                                # no instance at build time

def get_instructions(self):
    def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
        return f"Reviewing {ctx.deps.instance.name}."   # evaluated per request
    return _instructions
```

Return a plain string only for text that genuinely never changes — a persona, a
house style, a fixed rule. Anything derived from the instance, the clock, or the
database belongs in the callable.

Work that is expensive but stable can be done once and closed over, which is
what `DjangoModelCapability` does with the schema description:

```python
def get_instructions(self):
    schema = self.schema_description()          # computed once — fields do not
                                                # change between runs
    def _instructions(ctx):
        return f"{schema}\n\nCurrent values: {self.current_values(ctx.deps.instance)}"
    return _instructions
```

### Why instructions and not a system prompt

The current field values go out as *instructions*, not as a system prompt.

A system prompt is written into the message history and stays there. Because
this capability injects live field values, a system prompt would leave turn
one's values sitting in the context on turn five, contradicting the values that
are actually current. Instructions are re-sent fresh on each request and are
kept out of history, so there is only ever one set of values in play.

This is why `ModelAgent._system_prompts` is delivered as instructions too.

### What is cached and what is not

"Instructions are re-sent fresh on each request" is about *delivery*, not about
*content*. Every request carries a full instructions block, but only part of
that block is recomputed. The split trips people up, so it is worth being
precise.

Recomputed on every request, from the instance in `ctx.deps`:

- the current field values from `DjangoModelCapability`
- the current state and legal transitions from `DjangoFSMCapability`
- anything your own capability reads off `ctx.deps` inside its instructions
  callable

Computed once, when the agent is built, and then frozen for the life of that
agent object:

- `_instructions` and `_system_prompts`
- the rendered `_instructions_template`
- every `@ModelAgent.instructions` and `@ModelAgent.system_prompt` method
- the schema description (which fields exist and their types)

The reason is that `build_agent()` evaluates those into plain strings and hands
them to the capabilities, and the first call to `run()` or `run_sync()` caches
the built agent on `_pydantic_agent`. Later runs reuse it and never call your
methods again.

So this does what it looks like:

```python
agent = PlaceAgent(place)
await agent.run("Publish it.")        # tool sets is_published = True
await agent.run("Is it live yet?")    # sees is_published=True in current values
```

And this does not:

```python
class PlaceAgent(ModelAgent):
    model = Place

    @ModelAgent.instructions
    def tone(self) -> str:
        # Evaluated once. Still says "draft" on turn five, even after the
        # instance has been published.
        return "Be careful, this is live." if self.instance.is_published else "It's a draft."
```

If instructions genuinely need to track something that changes mid-conversation,
put it in a capability and read it off `ctx.deps` inside the instructions
callable — that is exactly what `DjangoModelCapability` does with field values.
See [Configuration in `__init__`, instance from `ctx.deps`](#configuration-in-__init__-instance-from-ctxdeps).

Two smaller footguns in the same area:

**The instance is not re-read from the database.** Fresh values come from the
in-memory `instance`, so a row another process updated will not show up. Call
`agent.refresh_instance()` before a run when that matters.

**Django caches compiled templates.** Under `DEBUG = False` the cached template
loader is in play, so editing an `_instructions_template` file does not take
effect until the process restarts. That is separate from the per-agent caching
above, and it bites in production, not in `runserver`.

To pick up changed static instructions, build a new agent:

```python
agent = PlaceAgent(place)      # cheap — the model schema is rebuilt lazily
```

There is no supported invalidation hook. Setting `agent._pydantic_agent = None`
forces a rebuild and the tests do exactly that, but it is a private attribute
and constructing a new agent is the honest version.

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

## Writing a custom capability

Capabilities are plain Pydantic AI capabilities. Subclass `AbstractCapability`
and implement only the hooks you need — every one is optional.

Here is a complete capability that tells the agent whether the instance still
needs a description:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from django_model_agent import ModelAgentContext


class DescriptionCoachCapability(AbstractCapability[ModelAgentContext]):   # (1)
    """Nudge the agent about the state of the description."""

    def __init__(self, *, min_words: int = 20):     # (2)
        self.min_words = min_words

    def get_instructions(self):                     # (3)
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            place = ctx.deps.instance               # (4)
            words = len(place.description.split())
            if words == 0:
                return "This place has no description. Offer to write one."
            if words < self.min_words:
                return f"The description is only {words} words. Suggest expanding it."
            return f"The description is {words} words, which is sufficient."

        return _instructions                        # (5)
```

1. Parameterise on `ModelAgentContext` — that is the deps type these agents use.
2. Configuration goes in `__init__`. **Never the model instance** — see below.
3. `get_instructions()` is the most common hook. It adds text to what the model
   sees each request.
4. The instance arrives per-run from `ctx.deps`, so this is always current.
5. Return the *callable*, not its result. A plain string would be evaluated once
   at construction and then go stale.

Use it like any other capability:

```python
class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description"]

    def get_extra_capabilities(self):
        return [DescriptionCoachCapability(min_words=30)]
```

### Configuration in `__init__`, instance from `ctx.deps`

This is the rule that keeps a capability reusable:

```python
# Wrong — binds the capability to one row forever
def __init__(self, *, place):
    self.place = place

# Right — read it per run
def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
    place = ctx.deps.instance
```

Capturing the instance means a separate agent per row, and values that go stale
the moment anything changes them.

### Adding tools

Return a toolset from `get_toolset()` and the capability carries its own tools:

```python
from pydantic_ai.toolsets import FunctionToolset


class GeocodeCapability(AbstractCapability[ModelAgentContext]):
    def get_toolset(self):
        toolset = FunctionToolset()

        @toolset.tool
        def lookup_coordinates(ctx: RunContext[ModelAgentContext]) -> str:
            """Look up the latitude and longitude for this place."""
            return geocode(ctx.deps.instance.address)

        return toolset
```

To wrap existing `ModelTool` classes instead, use
[`model_tools_to_toolset()`](reference.md#capabilities).

### Running code around a run

`before_run` and `after_run` fire either side of the run. Add `for_run()` when
you keep state, so two concurrent runs cannot see each other's:

```python
class TimingCapability(AbstractCapability[ModelAgentContext]):
    async def for_run(self, ctx):
        return TimingCapability()          # fresh instance per run

    async def before_run(self, ctx):
        self._started = time.monotonic()

    async def after_run(self, ctx, *, result):
        elapsed = time.monotonic() - self._started
        logger.info("Run took %.2fs", elapsed)
        return result                      # always return the result
```

!!! warning "Database access in hooks"

    `before_run` and `after_run` are async; Django's ORM is not. Wrap ORM calls
    in `asgiref.sync.sync_to_async`, and keep `get_instructions()` callables
    free of queries — have them read what a hook already loaded.

    ```python
    from asgiref.sync import sync_to_async

    async def before_run(self, ctx):
        self._row = await sync_to_async(MyModel.objects.get)(pk=ctx.deps.instance.pk)
    ```

!!! tip "More examples"

    The [cookbook](cookbook.md) has capabilities in context — scoping an agent
    to a tenant, persisting an audit trail, and adding memory.

See the [Pydantic AI capabilities docs](https://ai.pydantic.dev/capabilities/overview/)
for the full set of hooks.
