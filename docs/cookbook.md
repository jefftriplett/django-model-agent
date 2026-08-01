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

## Get structured data back instead of prose

By default `result.output` is a string. Declare an `output_type` and it becomes
a validated Pydantic model instead:

```python
from pydantic import BaseModel, Field


class PlaceReview(BaseModel):
    summary: str
    quality_score: int = Field(ge=1, le=5)
    missing_fields: list[str] = []
    should_publish: bool


class PlaceReviewAgent(ModelAgent):
    model = Place
    fields = ["name", "address", "phone", "description"]
    output_type = PlaceReview
    _system_prompts = "Review this place listing for completeness."


result = await PlaceReviewAgent(place).run("Review this listing.")

result.output.quality_score   # 4
result.output.missing_fields  # ["phone"]
```

The model is *forced* to return that shape — pydantic-ai validates the response
and retries if it does not match, so you never parse prose or handle a stray
"Sure! Here's the review:" preamble.

Set it per agent or per run when it varies:

```python
agent = PlaceReviewAgent(place, output_type=ShortSummary)   # per agent
result = await agent.run("Summarise.", output_type=ShortSummary)  # per run
```

### Storing the result

`result.output` is a normal Pydantic model, so `model_dump()` gives you a dict
ready for the ORM:

```python
review = result.output

place.description = review.summary
place.quality_score = review.quality_score
place.save(update_fields=["description", "quality_score"])
```

Into a `JSONField`, keeping the whole payload:

```python
class Place(models.Model):
    ...
    last_review = models.JSONField(null=True, blank=True)

place.last_review = review.model_dump(mode="json")   # (1)
place.save(update_fields=["last_review"])
```

1. `mode="json"` matters — it converts `datetime`, `Decimal`, and `UUID` into
   JSON-safe values. Plain `model_dump()` leaves Python objects that a
   `JSONField` cannot serialise.

Into a related model, one row per run:

```python
class PlaceReviewRecord(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE, related_name="reviews")
    summary = models.TextField()
    quality_score = models.PositiveSmallIntegerField()
    missing_fields = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)


PlaceReviewRecord.objects.create(place=place, **review.model_dump())
```

That last form works only while the field names line up. Once they drift, map
explicitly rather than renaming your schema to match the table — the schema is
what the model sees, and it should read well to the model first.

### Validate before you trust it

Validation guarantees the *shape*, not that the content is sensible. Keep your
own checks for anything that matters:

```python
review = result.output

if review.should_publish and place.state == "draft":
    if not place.address:
        raise ValueError("Refusing to publish without an address")
    place.publish()
    place.save()
```

Constrain what you can in the schema itself — `Field(ge=1, le=5)` above means an
out-of-range score never reaches your code.

### Unions when the answer varies

Let the model pick a shape by using a union, then branch on what came back:

```python
class NeedsWork(BaseModel):
    problems: list[str]

class ReadyToPublish(BaseModel):
    confidence: float

class PlaceAgent(ModelAgent):
    model = Place
    output_type = NeedsWork | ReadyToPublish

match (await PlaceAgent(place).run("Assess this listing.")).output:
    case ReadyToPublish(confidence=c) if c > 0.8:
        place.publish()
    case NeedsWork(problems=problems):
        place.notes = "\n".join(problems)
place.save()
```

See pydantic-ai's [output documentation](https://ai.pydantic.dev/output/) for
`ToolOutput`, `NativeOutput`, and `PromptedOutput`, which control *how* the
model is asked to produce the structure.

## Write prompts as Django templates

Once a prompt has conditionals in it, building it with string concatenation gets
ugly fast. Point `_instructions_template` at a template instead:

```python
class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description", "state"]
    _instructions_template = "agents/place_instructions.txt"
```

```django
{# templates/agents/place_instructions.txt #}
You are reviewing {{ instance.name }}.

{% if not instance.description %}
This place has no description. Offer to write one.
{% elif instance.description|wordcount < 20 %}
The description is very short — suggest expanding it.
{% endif %}

{% if instance.state == "draft" %}
This listing is not public yet. Focus on what still needs filling in.
{% else %}
This listing is live. Be conservative about suggesting changes.
{% endif %}
```

The template receives `instance` and `schema` in its context. Everything Django
templates already give you works — `{% if %}`, filters like `wordcount`,
`{% include %}` for shared fragments, and `{% for %}` over related objects.

Use a `.txt` template, not `.html`. Django's autoescaping will turn an apostrophe
in a model field into `&#x27;` and the model will read it literally.

A missing template or a syntax error inside one does not raise. It logs a warning
and renders as an empty string, so a typo in the path costs you the whole prompt
and the agent still answers — just worse. Pin it down with a test:

```python
def test_instructions_render(place):
    assert place.name in PlaceAgent(place).get_instructions()
```

!!! note "The template is re-rendered on every request"

    Conditionals on instance state stay correct as the instance changes, but the
    render cost is paid per model request — and one `run()` may make several.
    Keep templates cheap; see
    [when instructions are re-evaluated](capabilities.md#when-instructions-are-re-evaluated).

For prompts that need data beyond the instance, render it yourself and pass it in:

```python
from django.template.loader import render_to_string


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "description"]


def agent_for(place, *, house_style):
    return PlaceAgent(
        place,
        instructions=render_to_string(
            "agents/place_instructions.txt",
            {"instance": place, "house_style": house_style,
             "examples": PlaceExample.objects.filter(approved=True)[:3]},
        ),
    )
```

This is also how you keep prompts editable by non-developers: store the template
in the database, render it with `Template(...).render(Context(...))`, and pass
the result as `instructions`.

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
