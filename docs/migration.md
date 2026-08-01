# Migration guide

Recent releases moved django-model-agent onto Pydantic AI's
[capabilities](capabilities.md) system. Existing agents keep working — this
page covers the two things that changed underneath, and what to move to.

## Nothing to do for most agents

The declarative API is unchanged:

```python
class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "address"]
    _system_prompts = "You are a helpful assistant."
    tools = [GetPlaceInfoTool, UpdateDescriptionTool]

agent = PlaceAgent(place)
result = await agent.run("What is the address?")
```

`build_agent()` now composes capabilities instead of assembling everything
itself, but the classes you write are the same.

## System prompts are now instructions

`_system_prompts` is no longer passed to the agent as a system prompt. It is
combined with `_instructions` and sent as instructions.

**This is a correctness fix.** The agent injects the model's *current* field
values. A system prompt is written into the message history and stays there, so
the values captured on the first turn would still be in context on the fifth,
contradicting the values that are actually current. Instructions are re-sent
fresh on each request and kept out of history.

`_system_prompts`, `_instructions`, and the `@ModelAgent.system_prompt` and
`@ModelAgent.instructions` decorators all still work. Only the delivery
channel changed.

If you were reading system prompts back off the message history, that content
now lives on the request's `instructions` instead:

```python
result = await agent.run("...")

# Before: system prompt appeared as a message part.
# Now:
instructions = [
    m.instructions for m in result.all_messages() if getattr(m, "instructions", None)
]
```

A future release will merge `_system_prompts` and `_instructions` into a single
attribute. Both are supported until then.

## AgentMemoryMixin is deprecated

Use [`DjangoMemoryCapability`](capabilities.md#djangomemorycapability) instead.
Subclassing `AgentMemoryMixin` now raises a `DeprecationWarning`.

The mixin required multiple inheritance and manual saves:

```python
class PlaceAgent(AgentMemoryMixin, ModelAgent):
    model = Place

agent = PlaceAgent(place)
result = await agent.run("...")
agent.memory.append_to_history("user", "...")
agent.save_memory()
```

The capability composes instead, and loads and saves around each run for you:

```python
class PlaceAgent(ModelAgent):
    model = Place

    def get_extra_capabilities(self):
        return [DjangoMemoryCapability(max_history=50)]

agent = PlaceAgent(place)
result = await agent.run("...")   # memory loaded and saved automatically
```

The `AgentMemory` model, its manager, and the stored data are unchanged, so
there is no data migration — memory written through the mixin is read by the
capability.

To keep using the mixin for now, silence the warning:

```python
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*AgentMemoryMixin.*"
)
```

## Run the memory migration

`AgentMemory` was not registered with Django in earlier versions, so its table
was never created even though the docs told you to migrate. That is fixed. If
you use memory, apply the migration:

```console
python manage.py migrate django_model_agent
```

## Pydantic AI 2.22 is now required

The capabilities API does not exist in earlier versions. The declared minimum
was previously `0.2`, which could not actually work.

```toml
dependencies = ["pydantic-ai>=2.22"]
```
