# Tools

django-model-agent provides a hierarchy of tool base classes for building
tools that operate on Django model instances. Tools integrate with pydantic-ai
through `build_agent()`, which converts them into pydantic-ai compatible tool
functions automatically.

## Tools vs capabilities

A **tool** is one callable the model can invoke. It takes arguments, does
something, and returns a result. That is the whole contract.

A [**capability**](capabilities.md) is a bundle of agent behaviour. It can carry
tools, but it can also contribute instructions, set model settings, and hook the
run lifecycle.

Reach for a capability instead of a tool when the behaviour is not really about
one function:

| You want to… | Use |
|---|---|
| Let the model look something up or change a field | a tool |
| Tell the model something about the instance | a capability (instructions) |
| Show or hide tools based on state | a capability (`prepare_tools`) |
| Do something before or after every run | a capability (`before_run` / `after_run`) |
| Keep state across a run | a capability (`for_run`) |

`DjangoFSMCapability` is the clearest case. It hides tools the current state
forbids *and* tells the agent which state it is in — a tool has nowhere to put
that second half.

Tools you write here are collected into a toolset by
[`DjangoModelCapability`](capabilities.md#djangomodelcapability), so the two
layers meet there.

## Tool hierarchy

```
ModelTool (abstract base)
├── ReadOnlyTool      — reads data, never modifies
├── UpdateTool        — captures state, applies changes, saves
└── DiffAwareUpdateTool — proposes changes for review before applying
```

## Writing a custom tool

A complete tool, start to finish. Subclass `ModelTool`, give it a `name` and a
`description`, and implement `execute()`:

```python
from django_model_agent.tools import ModelTool, ToolResult


class WordCountTool(ModelTool):
    name = "word_count"                                    # (1)
    description = "Count the words in this place's description"   # (2)

    def execute(self, **kwargs) -> ToolResult:             # (3)
        words = len(self.instance.description.split())     # (4)
        return ToolResult(                                 # (5)
            success=True,
            message=f"{words} words",
            data={"word_count": words},
        )
```

1. The name the model calls. Must be unique on the agent.
2. Shown to the model — this is how it decides when to call the tool, so write
   it for the model, not for you.
3. Always accept `**kwargs`; the model may send arguments you did not declare.
4. `self.instance` is the Django model instance. `self.agent` is the parent
   `ModelAgent`.
5. Return a `ToolResult`, not a bare string. `message` is what the model sees.

Register it on an agent:

```python
class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description"]
    tools = [WordCountTool]
```

That is enough to run:

```python
result = await PlaceAgent(place).run("How long is the description?")
```

### Taking arguments

Declare keyword arguments on `execute()` and the model will fill them in. Give
them defaults so a call that omits them still works:

```python
class TruncateDescriptionTool(ModelTool):
    name = "truncate_description"
    description = "Shorten the description to a maximum number of words"

    def execute(self, *, max_words: int = 50, **kwargs) -> ToolResult:
        words = self.instance.description.split()
        self.instance.description = " ".join(words[:max_words])
        self.instance.save(update_fields=["description"])
        return ToolResult(success=True, message=f"Truncated to {max_words} words")
```

### Picking a base class

`ModelTool` is the general case. For the two common shapes there are base
classes that remove the boilerplate:

- **[`ReadOnlyTool`](#readonlytool)** — implement `read()` returning a dict; it
  wraps the result for you
- **[`UpdateTool`](#updatetool)** — implement `update()` mutating the instance;
  it diffs the fields and saves for you

Use plain `ModelTool` when a tool neither simply reads nor simply writes — the
`WordCountTool` above computes something, and `TruncateDescriptionTool` needs to
control its own save.

!!! tip "More examples"

    The [cookbook](cookbook.md) has tools in context — gating them behind
    workflow state, proposing changes for human review, and testing them without
    calling an API.

## ModelTool

The base class for all tools. Provides context injection, state checking, and
both sync and async execution paths.

```python
from django_model_agent.tools import ModelTool, ToolResult

class CheckAvailabilityTool(ModelTool):
    name = "check_availability"
    description = "Check if this place is currently open"
    allowed_states = ["public", "featured"]

    def execute(self, **kwargs) -> ToolResult:
        place = self.instance
        is_open = place.check_hours()
        return ToolResult(
            success=True,
            message=f"{'Open' if is_open else 'Closed'}",
            data={"is_open": is_open},
        )
```

### Class attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the tool |
| `description` | `str` | Human-readable description shown to the AI |
| `requires_confirmation` | `bool` | Advisory flag; **not enforced** — see below (default `False`) |
| `allowed_states` | `list[str] \| None` | FSM states where this tool is allowed (`None` = all) |

!!! warning "`requires_confirmation` is not enforced"

    Nothing in the library reads this attribute. Setting it to `True` does not
    gate the write — `UpdateTool` saves regardless, and it defaults to `True`
    there. Treat it as documentation of intent only.

    To actually require approval, either use `preview=True` and apply the change
    yourself, or persist a proposal for review — see
    [Propose changes for human review](cookbook.md#propose-changes-for-human-review).

### State checking

If `allowed_states` is set and the model instance has a `state` attribute,
the tool automatically checks whether it can run:

```python
class PublishTool(ModelTool):
    name = "publish"
    description = "Publish this place"
    allowed_states = ["draft"]

    def execute(self, **kwargs) -> ToolResult:
        self.instance.state = "public"
        self.instance.save()
        return ToolResult(success=True, message="Published")
```

Calling the tool when the instance is in the wrong state returns a failure
result without executing:

```python
tool = PublishTool(context)
result = tool(action="publish")  # Fails if state != "draft"
```

This check always runs, so `allowed_states` is enforced no matter how the tool
is invoked. Adding
[`DjangoFSMCapability`](capabilities.md#djangofsmcapability) goes a step
further and hides the tool from the model entirely while it is unavailable, so
the model never spends tokens on a call that can only be refused.

### Properties

`instance`
:   The Django model instance the tool operates on.

`agent`
:   The parent `ModelAgent` instance.

## ToolResult

Every tool execution returns a `ToolResult`:

```python
from django_model_agent.tools import ToolResult

result = ToolResult(
    success=True,
    message="Hours updated to: 9am-5pm",
    data={"hours": "9am-5pm"},
    changes={"hours": {"before": "10am-6pm", "after": "9am-5pm"}},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the tool executed successfully |
| `message` | `str` | Human-readable result message |
| `data` | `dict \| None` | Optional structured data |
| `changes` | `dict` | Field changes made (for audit/diff) |

## ReadOnlyTool

For tools that only read data and never modify the instance:

```python
from django_model_agent.tools import ReadOnlyTool

class GetContactInfoTool(ReadOnlyTool):
    name = "get_contact_info"
    description = "Get the contact information for this place"

    def read(self, **kwargs) -> dict:
        return {
            "phone": self.instance.phone,
            "website": self.instance.website,
            "address": self.instance.address,
        }
```

Override `read()` instead of `execute()`. The base class wraps the return
value in a successful `ToolResult` automatically.

## UpdateTool

For tools that modify the instance. Captures state before and after the
update, computes a diff, and calls `save()` automatically:

```python
from django_model_agent.tools import UpdateTool

class UpdateDescriptionTool(UpdateTool):
    name = "update_description"
    description = "Update the place description"
    allowed_states = ["draft", "public"]

    def update(self, *, description: str, **kwargs) -> None:
        self.instance.description = description
```

Override `update()` instead of `execute()`. Do not call `save()` — the base
class handles that after computing the diff.

The `changes` field on the returned `ToolResult` contains the diff:

```python
result = tool.execute(description="New description")
result.changes
# {'description': {'before': 'Old text', 'after': 'New description'}}
```

### Preview mode

Pass `preview=True` to see what would change without saving:

```python
result = tool.execute(description="New text", preview=True)
# Instance is modified in memory but not saved to database
```

## DiffAwareUpdateTool

For multi-agent workflows where one agent proposes changes and another
reviews them:

```python
from django_model_agent.tools import DiffAwareUpdateTool, ToolResult

class ProposeUrlChangeTool(DiffAwareUpdateTool):
    name = "propose_url_change"
    description = "Propose a URL update for review"

    def execute(self, *, field: str, url: str, reason: str = "", **kwargs) -> ToolResult:
        self.propose_change(field, url, reason)
        return ToolResult(
            success=True,
            message=f"Proposed change to {field}. Awaiting approval.",
        )
```

### Workflow

```python
# Agent 1 proposes changes
tool = ProposeUrlChangeTool(context)
tool.execute(field="website", url="https://new-site.com", reason="URL updated")

# Review pending changes
pending = tool.get_pending_changes()
summary = tool.get_diff_summary()

# Agent 2 or human approves/rejects
for change in pending:
    change.approve()  # or change.reject()

# Apply approved changes
applied = tool.apply_approved_changes()
```

### ProposedChange

Each proposed change is a `ProposedChange` object:

| Field | Description |
|-------|-------------|
| `instance` | The model instance |
| `field_name` | Name of the field being changed |
| `old_value` | Current value |
| `new_value` | Proposed new value |
| `reason` | Why this change is being proposed |
| `approved` | `True`, `False`, or `None` (pending) |

## Decorated tools

Instead of subclassing `ModelTool`, you can register tools with the
`@ModelAgent.tool` decorator:

```python
class RestaurantAgent(ModelAgent):
    model = Restaurant
    fields = ["name", "hours"]

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

The method's docstring becomes the tool description, and the method signature
defines the tool's parameters. Both `ModelTool` subclasses and decorated tools
can be used together on the same agent.

## How tools are converted for pydantic-ai

Tools are turned into a pydantic-ai toolset by
[`DjangoModelCapability`](capabilities.md#djangomodelcapability), which
`build_agent()` composes for you:

- **`ModelTool` subclasses** are wrapped as `pydantic_ai.Tool` objects with
  `takes_ctx=True`. The tool is constructed per call from `ctx.deps`, so it
  always sees the instance for the current run.
- **Decorated methods** are wrapped the same way, with their original
  signatures preserved and the agent resolved from `ctx.deps.agent`.

Both kinds take the run context rather than closing over a particular instance.
That is what lets one agent serve every row of the table instead of needing to
be rebuilt per instance — see
[Capabilities](capabilities.md#using-them-with-a-plain-agent).
