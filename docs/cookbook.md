# Cookbook

Task-oriented recipes. Each one is self-contained — skim the headings and take
what you need.

## Run one agent over many rows

Capabilities take a model *class*, not an instance, so a single agent handles
every row. Build it once outside the loop:

```python
from pydantic_ai import Agent

from django_model_agent import ModelAgentContext
from django_model_agent.capabilities import DjangoModelCapability

agent = Agent(
    "openai:gpt-4o",
    deps_type=ModelAgentContext,
    capabilities=[
        DjangoModelCapability(
            model_class=Place,
            fields=["name", "address", "description"],
            instructions="Write a one-line summary of this place.",
        )
    ],
)

for place in Place.objects.filter(state="draft"):
    result = await agent.run(
        "Summarise this place.",
        deps=ModelAgentContext(instance=place, agent=None),
    )
    place.summary = result.output
    place.save(update_fields=["summary"])
```

Rebuilding the agent inside the loop would work, but it re-derives the schema
and re-registers every tool on each iteration for no benefit — nothing in the
agent depends on which row you are looking at.

Using `ModelAgent`, construct one per instance as usual; the cost is small, and
you get the declarative class back:

```python
for place in Place.objects.filter(state="draft"):
    result = await PlaceAgent(place).run("Summarise this place.")
```

## Expose different fields to different roles

`_field_sets` names groups of fields; pick one at construction time:

```python
class PlaceAgent(ModelAgent):
    model = Place

    _field_sets = {
        "public": ["name", "address", "phone"],
        "staff": ["name", "address", "phone", "notes", "internal_id"],
        "admin": None,          # None means every field
    }

    _system_prompts = "You manage place records."


def agent_for(user, place):
    if user.is_superuser:
        return PlaceAgent(place, field_set="admin")
    if user.is_staff:
        return PlaceAgent(place, field_set="staff")
    return PlaceAgent(place, field_set="public")
```

Fields outside the chosen set never reach the model — they are absent from both
the schema description and the current values.

## Record an audit trail in your own model

`DjangoAuditCapability` hands each run's changes to a callback. Persist them
however you like:

```python
from django_model_agent.capabilities import DjangoAuditCapability


class AgentAudit(models.Model):
    object_pk = models.CharField(max_length=64)
    model_name = models.CharField(max_length=64)
    prompt = models.TextField()
    changes = models.JSONField(default=dict)
    tools_used = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)


def persist(record):
    if not record.changed:
        return                      # nothing happened; skip the row
    AgentAudit.objects.create(
        object_pk=str(record.instance_pk),
        model_name=record.model_class,
        prompt=record.prompt,
        changes=record.field_changes,
        tools_used=[call["name"] for call in record.tool_calls],
    )


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description"]
    tools = [UpdateDescriptionTool]

    def get_extra_capabilities(self):
        return [DjangoAuditCapability(log_to="callback", callback=persist)]
```

Use the callback rather than reading `capability.record`: every run gets its own
copy of the capability, so the instance you constructed never receives the
record.

## Gate tools behind workflow state

Declare where each tool is legal, and add the FSM capability so the model never
sees the ones it cannot use:

```python
from django_model_agent.capabilities import DjangoFSMCapability


class PublishTool(UpdateTool):
    name = "publish"
    description = "Publish this place"
    allowed_states = ["draft"]

    def update(self, **kwargs) -> None:
        self.instance.state = "public"


class CloseTool(UpdateTool):
    name = "close"
    description = "Mark this place permanently closed"
    allowed_states = ["public", "featured"]

    def update(self, **kwargs) -> None:
        self.instance.state = "closed"


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "state"]
    tools = [PublishTool, CloseTool]
```

`ModelAgent` adds `DjangoFSMCapability` automatically when any tool restricts by
state. To describe the transitions as well, add it yourself:

```python
    def get_extra_capabilities(self):
        return [
            DjangoFSMCapability(
                tools=self.tools,
                transitions={
                    "draft": ["public"],
                    "public": ["featured", "closed"],
                    "featured": ["public"],
                    "closed": [],
                },
            )
        ]
```

Filtering is an optimisation, not a security boundary. `ModelTool.check_allowed()`
still runs on execution, so a tool called some other way is still refused.

## Propose changes for human review

`DiffAwareUpdateTool` collects proposals on the tool instance, which works when
you drive the tool yourself:

```python
tool = ProposeUrlChangeTool(ModelAgentContext(instance=place, agent=None))
tool.execute(field="website", url="https://new-site.com", reason="moved")

for change in tool.get_pending_changes():
    change.approve()
tool.apply_approved_changes()
```

!!! warning "In-memory proposals do not survive an agent run"

    When a tool runs through an agent it is constructed per call, so you cannot
    reach the instance afterwards to read `proposed_changes`. Anything you need
    after the run must be persisted by the tool itself.

For agent-driven review, write proposals to a model:

```python
class ProposedEdit(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    field_name = models.CharField(max_length=64)
    new_value = models.TextField()
    reason = models.TextField(blank=True)
    approved = models.BooleanField(null=True)


class ProposeUrlTool(ModelTool):
    name = "propose_url"
    description = "Propose a website URL change for human review"

    def execute(self, *, url: str, reason: str = "", **kwargs) -> ToolResult:
        ProposedEdit.objects.create(
            place=self.instance, field_name="website",
            new_value=url, reason=reason,
        )
        return ToolResult(success=True, message="Proposed; awaiting review.")
```

Then approve out of band:

```python
for edit in ProposedEdit.objects.filter(place=place, approved=None):
    edit.approved = True
    edit.save()
    setattr(edit.place, edit.field_name, edit.new_value)
    edit.place.save()
```

## Test agents without calling an API

Use pydantic-ai's `TestModel`. Build the agent, attach it, then override:

```python
from pydantic_ai.models.test import TestModel


def test_agent_runs(place):
    agent = PlaceAgent(place)
    pai = agent.build_agent()
    agent._pydantic_agent = pai        # so run_sync uses this one

    with pai.override(model=TestModel()):
        result = agent.run_sync("What is the address?")

    assert result.output is not None
```

Assigning `_pydantic_agent` matters: `run_sync()` lazily builds its own agent on
first use, which would not carry the override.

`TestModel` calls every tool it is offered, which makes it a good way to assert
what the model could see:

```python
def tools_offered(result):
    return {
        part.tool_name
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if type(part).__name__ == "ToolCallPart"
    }

assert "publish" in tools_offered(result)
assert "close" not in tools_offered(result)
```

### Asserting on instructions

Field values are sent as instructions, which are deliberately kept out of
message history. Read them from the requests:

```python
def instructions_sent(result):
    return "\n".join(
        m.instructions
        for m in result.all_messages()
        if getattr(m, "instructions", None)
    )

assert place.name in instructions_sent(result)
```

### Database access in tests

Tools that save, and `DjangoMemoryCapability`, reach the ORM from a different
thread than the test. Use a real transaction or they will deadlock:

```python
pytestmark = pytest.mark.django_db(transaction=True)
```

## Scope an agent to a tenant

Read per-run values from `ctx.deps` rather than capturing them at construction,
so the capability stays reusable:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability


class TenantCapability(AbstractCapability[ModelAgentContext]):
    """Tell the agent which tenant it is acting for."""

    def get_instructions(self):
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            tenant = ctx.deps.instance.tenant
            return (
                f"You are acting for {tenant.name}. "
                f"Never reference records belonging to another tenant."
            )

        return _instructions
```

Capturing `tenant` in `__init__` would bind the agent to whichever tenant
happened to be first — the bug this pattern avoids.

## Give an agent memory across conversations

```python
from django_model_agent.capabilities import DjangoMemoryCapability


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description"]

    def get_extra_capabilities(self):
        return [DjangoMemoryCapability(max_history=50)]
```

Past turns are replayed as instructions and the new turn is saved when the run
finishes. Requires `django_model_agent` in `INSTALLED_APPS` and its migration
applied.

To record turns without feeding them back — useful for a transcript you only
want to read later:

```python
DjangoMemoryCapability(include_history=False)
```

Unsaved instances are skipped, since there is no primary key to key memory to.
