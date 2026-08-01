# Memory

django-model-agent provides persistent memory storage tied to Django model
instances via the contenttypes framework. This lets agents remember context
across conversations about specific entities.

## Setup

Add `django_model_agent` to `INSTALLED_APPS` and run migrations:

```python
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    # ...
    "django_model_agent",
]
```

```console
python manage.py migrate django_model_agent
```

## Recommended: DjangoMemoryCapability

The simplest way to use memory is
[`DjangoMemoryCapability`](capabilities.md#djangomemorycapability), which loads
memory before each run and saves it after:

```python
from django_model_agent import ModelAgent
from django_model_agent.capabilities import DjangoMemoryCapability


class PlaceAgent(ModelAgent):
    model = Place
    fields = ["name", "address"]

    def get_extra_capabilities(self):
        return [DjangoMemoryCapability(max_history=50)]
```

Past turns are replayed automatically and the new turn is saved when the run
finishes.

They are replayed as real pydantic-ai messages, so tool calls and their results
carry into the next turn — the model can see it already looked something up,
rather than reading a summary that says so. The messages live on
`AgentMemory.data["messages"]` in pydantic-ai's own serialisation format.

`AgentMemoryMixin`, documented below, is deprecated in favour of this. See the
[migration guide](migration.md#agentmemorymixin-is-deprecated).

## AgentMemory model

`AgentMemory` uses a `GenericForeignKey` to attach a JSON blob to any Django
model instance:

```python
from django_model_agent.memory import AgentMemory

# Store memory for a restaurant
memory, created = AgentMemory.objects.get_or_create_for(restaurant)
memory.set("last_question", "hours")
memory.set("conversation_count", 1)
memory.save()

# Later, retrieve it
memory = AgentMemory.objects.get_for(restaurant)
memory.get("last_question")  # "hours"
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `content_type` | `ForeignKey` | The type of model this memory is attached to |
| `object_id` | `PositiveIntegerField` | The primary key of the model instance |
| `content_object` | `GenericForeignKey` | The model instance |
| `data` | `JSONField` | Arbitrary JSON data storing agent memory/state |
| `created_at` | `DateTimeField` | When the memory was created |
| `updated_at` | `DateTimeField` | When the memory was last modified |

### Manager methods

`AgentMemory.objects.get_for(instance)`
:   Get the memory record for a model instance. Returns `None` if not found.
    Raises `ValueError` if the instance has not been saved.

`AgentMemory.objects.get_or_create_for(instance, defaults=None)`
:   Get or create memory for a model instance. Returns a `(memory, created)`
    tuple. Raises `ValueError` if the instance has not been saved.

`AgentMemory.objects.filter_for_model(model_class)`
:   Get all memory records for a specific model type.

### Convenience methods

```python
memory.get("key", default=None)    # Get a value from memory data
memory.set("key", "value")         # Set a value (doesn't save)
memory.update(key1="a", key2="b")  # Update multiple values (doesn't save)
memory.clear()                      # Clear all memory data (doesn't save)
```

All mutating methods require an explicit `memory.save()` call.

### Conversation history

`AgentMemory` includes built-in support for tracking conversation history:

```python
memory.append_to_history("user", "What are the hours?")
memory.append_to_history("assistant", "We're open 9am to 5pm.")

history = memory.get_history()
# [{"role": "user", "content": "What are the hours?"}, ...]
```

History is automatically trimmed to `max_history` entries (default 100):

```python
memory.append_to_history("user", "Hello", max_history=50)
```

## AgentMemoryMixin

Add persistent memory to any `ModelAgent` subclass:

```python
from django_model_agent import ModelAgent
from django_model_agent.memory import AgentMemoryMixin

class RestaurantAgent(AgentMemoryMixin, ModelAgent):
    model = Restaurant
    fields = ["name", "hours"]
    _system_prompts = "You help with restaurant information."
```

The mixin adds:

`agent.memory`
:   Property that lazily loads or creates the `AgentMemory` for the instance.

`agent.load_memory()`
:   Load and return the memory data dict.

`agent.save_memory()`
:   Save the current memory state to the database.

`agent.get_memory_context()`
:   Format memory as a string for inclusion in prompts.

### Example

```python
restaurant = Restaurant.objects.get(pk=123)
agent = RestaurantAgent(restaurant)

# Store context from this conversation
agent.memory.set("last_topic", "hours")
agent.memory.append_to_history("user", "What are the hours?")
agent.save_memory()

# Next conversation — memory persists
agent = RestaurantAgent(restaurant)
context = agent.get_memory_context()
# "Previous context:\n  last_topic: hours\n  Conversation history: 1 messages"
```
