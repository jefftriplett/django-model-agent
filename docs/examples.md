# Examples

Four agents built around a `Place` model — a venue listing with a name, address,
description, opening hours, and a `state` field for its publication workflow.

Each shows a different shape of problem. The [cookbook](cookbook.md) covers
individual techniques; this page shows them working together.

## Will it rain at the venue today?

Multiple tools the model has to call *in order*: it cannot fetch a forecast
without coordinates, and it cannot get coordinates without the address.

The Django twist is that the address is already on the model, so unlike a
general weather agent there is nothing to ask the user for — the agent starts
with the location and works outward.

```python
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from django_model_agent import ModelAgent
from django_model_agent.tools import ModelTool, ReadOnlyTool, ToolResult


class GeocodeTool(ReadOnlyTool):
    name: ClassVar[str] = "geocode_address"
    description: ClassVar[str] = "Look up latitude and longitude for this place's address"

    def read(self, **kwargs: Any) -> dict[str, Any]:
        if not self.instance.address:
            return {"error": "This place has no address on file."}

        response = httpx.get(
            "https://geocode.maps.co/search",
            params={"q": self.instance.address, "api_key": settings.GEO_API_KEY},
            timeout=10,
        )
        hits = response.json()
        if not hits:
            return {"error": f"Could not geocode {self.instance.address!r}."}
        return {"lat": float(hits[0]["lat"]), "lng": float(hits[0]["lon"])}


class ForecastTool(ModelTool):
    name: ClassVar[str] = "get_forecast"
    description: ClassVar[str] = "Get today's forecast for a latitude and longitude"

    def execute(self, *, lat: float = 0.0, lng: float = 0.0, **kwargs: Any) -> ToolResult:
        response = httpx.get(
            "https://api.tomorrow.io/v4/weather/forecast",
            params={"location": f"{lat},{lng}", "timesteps": "1d",
                    "apikey": settings.WEATHER_API_KEY},
            timeout=10,
        )
        today = response.json()["timelines"]["daily"][0]["values"]
        return ToolResult(
            success=True,
            message=f"{today['precipitationProbabilityAvg']}% chance of rain",
            data={
                "precipitation_chance": today["precipitationProbabilityAvg"],
                "high_c": today["temperatureMax"],
                "low_c": today["temperatureMin"],
            },
        )


class WeatherAnswer(BaseModel):
    will_rain: bool
    chance_percent: int = Field(ge=0, le=100)
    advice: str


class VenueWeatherAgent(ModelAgent):
    model = Place
    fields = ["name", "address"]
    tools = [GeocodeTool, ForecastTool]
    output_type = WeatherAnswer
    _system_prompts = (
        "You advise venues about weather. Geocode the address first, then fetch "
        "the forecast for those coordinates, then answer. Never guess coordinates."
    )
```

```python
result = await VenueWeatherAgent(place).run("Is it going to rain today?")

if result.output.will_rain:
    notify_staff(
        f"{place.name}: {result.output.chance_percent}% rain — {result.output.advice}"
    )
```

Two things make this work:

- **Tool arguments are declared with defaults.** `ForecastTool.execute()` takes
  `lat` and `lng`, which is how the model passes the geocoder's output into the
  forecast call. Give them defaults so a call that omits one still runs.
- **The output is a schema, not prose.** `will_rain` is a real boolean, so the
  branch above is a normal Python `if` rather than string matching.

Returning an `error` key from a tool rather than raising lets the model recover —
it can tell the user the address is missing instead of the run failing.

## Extracting a listing from unstructured text

Someone pastes in a venue's details from an email or a PDF. Turn that into
typed fields ready for the ORM.

```python
from pydantic import BaseModel, Field


class ExtractedListing(BaseModel):
    name: str
    address: str | None = None
    phone: str | None = None
    cuisine: str | None = None
    description: str = Field(description="One or two sentences, neutral tone")
    confidence: float = Field(ge=0, le=1, description="How sure you are overall")


class ListingExtractor(ModelAgent):
    model = Place
    fields = ["name"]          # the agent barely needs the instance here
    output_type = ExtractedListing
    _system_prompts = (
        "Extract venue details from the text you are given. Leave a field null "
        "rather than guessing. Do not invent a phone number or address."
    )
```

```python
raw = request.POST["pasted_text"]
result = await ListingExtractor(place).run(f"Extract the venue details:\n\n{raw}")
listing = result.output

if listing.confidence < 0.6:
    place.notes = f"Low-confidence extraction:\n{listing.model_dump_json(indent=2)}"
    place.state = "needs_review"
else:
    place.name = listing.name
    place.address = listing.address or place.address
    place.description = listing.description
place.save()
```

The `confidence` field is doing real work: it routes uncertain extractions to a
human instead of writing them straight to the table. `Field(description=...)`
is worth using — it becomes part of the schema the model sees, so it is a place
to say *"neutral tone"* or *"leave null rather than guessing"* per field.

## A chat interface over a model instance

Conversation across turns, streamed to the browser. Memory does the remembering,
so the view holds no session state.

```python
from django_model_agent import ModelAgent
from django_model_agent.capabilities import DjangoMemoryCapability


class PlaceChatAgent(ModelAgent):
    model = Place
    fields = ["name", "address", "phone", "description", "hours", "state"]
    tools = [UpdateDescriptionTool, UpdateContactInfoTool]
    _system_prompts = (
        "You help staff maintain this venue's listing. Answer questions about it "
        "and make edits when asked. Confirm what you changed."
    )

    def get_extra_capabilities(self):
        return [DjangoMemoryCapability(max_history=40)]
```

```python
import json

from django.http import StreamingHttpResponse


async def chat(request, pk):
    place = await Place.objects.aget(pk=pk)
    agent = PlaceChatAgent(place)
    message = request.POST["message"]

    async def stream():
        async with agent.run_stream(message) as result:
            async for chunk in result.stream_text(delta=True):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    response = StreamingHttpResponse(stream(), content_type="text/event-stream")
    response["X-Accel-Buffering"] = "no"
    return response
```

There is no conversation state in the view. `DjangoMemoryCapability` loads the
history before the run and saves it after, keyed to the `Place` — so the next
request, from any worker, continues where this one stopped.

Because history is stored as real messages, a follow-up like *"undo that"* works:
the model can see the tool call it made and what it returned, not a summary
saying it made one.

## A moderation queue

An agent reviews a draft listing and asks to publish it. Publishing is gated, so
the run stops and waits for a human.

```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelMessagesTypeAdapter

from django_model_agent.tools import UpdateTool


class PublishTool(UpdateTool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish this listing"
    allowed_states: ClassVar[list[str]] = ["draft"]
    requires_confirmation: ClassVar[bool] = True

    def update(self, **kwargs: Any) -> None:
        self.instance.state = "public"


class ModerationAgent(ModelAgent):
    model = Place
    fields = ["name", "address", "phone", "description", "state"]
    tools = [PublishTool, FlagForReviewTool]
    _system_prompts = (
        "Review this listing. Publish it only if the name, address, and "
        "description are all present and read professionally. Otherwise flag it "
        "and say what is missing."
    )
```

`allowed_states` means the model is not even offered `publish` unless the
listing is a draft. `requires_confirmation` means that even then, it only gets to
*ask*:

```python
result = await ModerationAgent(place).run("Review this listing.")

if isinstance(result.output, DeferredToolRequests):
    for call in result.output.approvals:
        ModerationRequest.objects.create(
            place=place,
            tool_call_id=call.tool_call_id,
            messages=ModelMessagesTypeAdapter.dump_python(
                result.all_messages(), mode="json"
            ),
        )
```

A reviewer approves it in the admin, and the run picks up where it left off:

```python
async def approve(request_id, *, approved: bool):
    pending = await ModerationRequest.objects.aget(pk=request_id)
    place = await Place.objects.aget(pk=pending.place_id)

    results = DeferredToolResults()
    results.approvals[pending.tool_call_id] = approved

    await ModerationAgent(place).run(
        message_history=ModelMessagesTypeAdapter.validate_python(pending.messages),
        deferred_tool_results=results,
    )
```

Three layers of control, each doing a different job:

| Layer | Stops |
|---|---|
| `allowed_states` | the tool being offered in the wrong state |
| `requires_confirmation` | it running without a human |
| `check_allowed()` | it running if reached some other way |

Add [`DjangoAuditCapability`](capabilities.md#djangoauditcapability) and every
decision is recorded with the fields it changed and the tokens it cost.

## Testing any of these

None of the examples above need an API key to test. Swap in `TestModel`:

```python
from pydantic_ai.models.test import TestModel


def test_weather_agent_calls_tools_in_order(place):
    agent = VenueWeatherAgent(place)
    pai = agent.build_agent()
    agent._pydantic_agent = pai

    with pai.override(model=TestModel()):
        result = agent.run_sync("Is it going to rain today?")

    called = [
        part.tool_name
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if type(part).__name__ == "ToolCallPart"
    ]
    assert called[:2] == ["geocode_address", "get_forecast"]
```

`TestModel` calls every tool it is offered, which makes it good for asserting
what an agent *could* do — and, with `allowed_states`, what it correctly could
not. See [testing agents](cookbook.md#test-agents-without-calling-an-api).
