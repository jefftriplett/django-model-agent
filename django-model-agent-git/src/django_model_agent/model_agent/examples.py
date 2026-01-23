"""
Example implementations of ModelAgent for the LFK.im project.

These examples show how to create agents bound to Django models,
demonstrating various features like field sets, tools, and memory.
"""

from __future__ import annotations

from typing import Any, ClassVar

from places.models import Place

from .base import ModelAgent
from .tools import DiffAwareUpdateTool, ModelTool, ReadOnlyTool, ToolResult, UpdateTool


# -----------------------------------------------------------------------------
# Tools for Place model
# -----------------------------------------------------------------------------


class GetPlaceInfoTool(ReadOnlyTool):
    """Tool to get comprehensive information about a place."""

    name: ClassVar[str] = "get_place_info"
    description: ClassVar[str] = (
        "Get detailed information about this place including name, address, "
        "contact info, delivery options, and current state."
    )

    def read(self, **kwargs: Any) -> dict[str, Any]:
        place: Place = self.instance
        return {
            "name": place.name,
            "address": place.address,
            "locality": place.locality,
            "region": place.region,
            "phone": place.phone,
            "website": place.website,
            "state": place.state,
            "description": place.description,
            "has_delivery": place.delivery,
            "has_takeout": place.takeout,
            "has_dinein": place.dinein,
            "has_curbside": place.curbside,
        }


class GetDeliveryOptionsTool(ReadOnlyTool):
    """Tool to get all delivery service links for a place."""

    name: ClassVar[str] = "get_delivery_options"
    description: ClassVar[str] = "Get all available delivery service URLs for this place."

    def read(self, **kwargs: Any) -> dict[str, Any]:
        place: Place = self.instance
        return {
            "doordash": place.doordash_url,
            "grubhub": place.grubhub_url,
            "ubereats": place.ubereats_url,
            "postmates": place.postmates_url,
            "eatstreet": place.eatstreet_url,
            "chownow": place.chownow_url,
            "menufy": place.menufy_url,
            "seamless": place.seamless_url,
            "delivery": place.delivery_url,  # Generic delivery URL
        }


class GetHoursTool(ReadOnlyTool):
    """Tool to get operating hours for a place."""

    name: ClassVar[str] = "get_hours"
    description: ClassVar[str] = "Get the operating hours for this place, organized by day of the week."

    def read(self, **kwargs: Any) -> dict[str, Any]:
        place: Place = self.instance
        hours_by_day = place.get_hours_by_day()

        # Convert to a more readable format
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        formatted_hours = {}

        for day_num, day_name in enumerate(day_names):
            if day_num in hours_by_day:
                formatted_hours[day_name] = [
                    {
                        "open": h.open_time.strftime("%I:%M %p"),
                        "close": h.close_time.strftime("%I:%M %p"),
                        "label": h.label or None,
                    }
                    for h in hours_by_day[day_num]
                ]
            else:
                formatted_hours[day_name] = None  # Closed

        return {"hours": formatted_hours}


class UpdateDescriptionTool(UpdateTool):
    """Tool to update the place description."""

    name: ClassVar[str] = "update_description"
    description: ClassVar[str] = "Update the description text for this place. " "Requires confirmation before saving."
    allowed_states: ClassVar[list[str]] = ["draft", "public", "featured"]

    def update(self, *, description: str, **kwargs: Any) -> None:
        self.instance.description = description


class UpdateContactInfoTool(UpdateTool):
    """Tool to update contact information."""

    name: ClassVar[str] = "update_contact_info"
    description: ClassVar[str] = "Update the contact information (phone, website, address) for this place."
    allowed_states: ClassVar[list[str]] = ["draft", "public", "featured"]

    def update(
        self,
        *,
        phone: str | None = None,
        website: str | None = None,
        address: str | None = None,
        **kwargs: Any,
    ) -> None:
        if phone is not None:
            self.instance.phone = phone
        if website is not None:
            self.instance.website = website
        if address is not None:
            self.instance.address = address


class ProposeDeliveryUrlTool(DiffAwareUpdateTool):
    """Tool that proposes delivery URL changes for review."""

    name: ClassVar[str] = "propose_delivery_url"
    description: ClassVar[str] = (
        "Propose updates to delivery service URLs. Changes will be queued " "for review before being applied."
    )

    def execute(
        self,
        *,
        service: str,
        url: str,
        reason: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        # Map service names to field names
        service_fields = {
            "doordash": "doordash_url",
            "grubhub": "grubhub_url",
            "ubereats": "ubereats_url",
            "postmates": "postmates_url",
            "eatstreet": "eatstreet_url",
            "chownow": "chownow_url",
            "menufy": "menufy_url",
            "seamless": "seamless_url",
            "delivery": "delivery_url",
        }

        if service.lower() not in service_fields:
            return ToolResult(
                success=False,
                message=f"Unknown service: {service}. Valid services: {', '.join(service_fields.keys())}",
            )

        field_name = service_fields[service.lower()]
        self.propose_change(field_name, url, reason)

        return ToolResult(
            success=True,
            message=f"Proposed change to {service} URL. Awaiting approval.",
            data={"field": field_name, "proposed_url": url},
        )


class ChangeStateTool(ModelTool):
    """Tool to transition place state using FSM."""

    name: ClassVar[str] = "change_state"
    description: ClassVar[str] = (
        "Change the state of this place. Available transitions depend on current state. "
        "States: draft, public, featured, closed, temp_closed"
    )
    requires_confirmation: ClassVar[bool] = True

    def execute(self, *, action: str, **kwargs: Any) -> ToolResult:
        place: Place = self.instance

        # Map action names to FSM transitions
        transitions = {
            "publish": place.publish,
            "feature": place.feature,
            "unfeature": place.unfeature,
            "close": place.close,
            "temp_close": place.temp_close,
            "reopen": place.reopen,
            "set_draft": place.set_draft,
        }

        if action not in transitions:
            return ToolResult(
                success=False,
                message=f"Unknown action: {action}. Valid actions: {', '.join(transitions.keys())}",
            )

        try:
            old_state = place.state
            transitions[action]()
            place.save()
            return ToolResult(
                success=True,
                message=f"State changed from '{old_state}' to '{place.state}'",
                changes={"state": {"before": old_state, "after": place.state}},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Cannot perform '{action}' from state '{place.state}': {e}",
            )


class FlagForReviewTool(ModelTool):
    """Tool to flag a place for human review."""

    name: ClassVar[str] = "flag_for_review"
    description: ClassVar[str] = (
        "Flag this place for human review with a note explaining why. "
        "Use this when information seems incorrect or needs verification."
    )

    def execute(self, *, reason: str, **kwargs: Any) -> ToolResult:
        place: Place = self.instance

        # Add to notes field
        existing_notes = place.notes or ""
        flag_note = f"\n\n[FLAGGED FOR REVIEW]: {reason}"
        place.notes = existing_notes + flag_note
        place.save()

        return ToolResult(
            success=True,
            message=f"Place flagged for review: {reason}",
        )


# -----------------------------------------------------------------------------
# PlaceAgent implementation
# -----------------------------------------------------------------------------


class PlaceAgent(ModelAgent):
    """
    AI agent for reasoning about and managing Place records.

    This agent can answer questions about restaurant information,
    help update records, and propose changes for review.

    Example:
        place = Place.objects.get(pk=123)
        agent = PlaceAgent(place)

        # Ask about hours
        result = await agent.run("What are the hours on Saturday?")

        # Propose a change
        result = await agent.run("Update the DoorDash URL to https://doordash.com/new")
    """

    model = Place

    fields: ClassVar[list[str]] = [
        "name",
        "slug",
        "address",
        "locality",
        "region",
        "phone",
        "website",
        "description",
        "state",
        "delivery",
        "takeout",
        "dinein",
        "curbside",
    ]

    field_sets: ClassVar[dict[str, list[str]]] = {
        "public": [
            "name",
            "address",
            "locality",
            "phone",
            "website",
            "description",
        ],
        "staff": [
            "name",
            "slug",
            "address",
            "locality",
            "region",
            "phone",
            "website",
            "description",
            "state",
            "notes",
            "delivery",
            "takeout",
            "dinein",
            "curbside",
        ],
        "admin": None,  # All fields
    }

    base_prompt: ClassVar[str] = """
You are a helpful assistant that manages restaurant and business information for LFK.im,
a local directory for Lawrence, Kansas.

Your primary responsibilities:
1. Answer questions about place information accurately using the available data
2. Help update records when given new information
3. Flag inconsistencies or issues for human review
4. Never invent or guess information that isn't in the data

When updating information:
- Always verify the change makes sense in context
- Use the appropriate tool for the type of change
- For significant changes, propose them for review rather than applying directly

Current place state affects what actions are available:
- draft: Can be edited freely, can be published
- public: Active listing, can be featured or closed
- featured: Highlighted listing, can be unfeatured or closed
- closed: Permanently closed, can be reopened
- temp_closed: Temporarily closed, can be reopened
"""

    tools: ClassVar[list[type[ModelTool]]] = [
        GetPlaceInfoTool,
        GetDeliveryOptionsTool,
        GetHoursTool,
        UpdateDescriptionTool,
        UpdateContactInfoTool,
        ProposeDeliveryUrlTool,
        ChangeStateTool,
        FlagForReviewTool,
    ]

    def __init__(self, instance: Place, *, field_set: str | None = None) -> None:
        super().__init__(instance, field_set=field_set)

    def get_system_prompt(self) -> str:
        """Enhance system prompt with current place context."""
        base_prompt = super().get_system_prompt()

        # Add current state context
        state_info = f"\n\nCurrent place: {self.instance.name}"
        state_info += f"\nCurrent state: {self.instance.state}"

        if self.instance.neighborhood:
            state_info += f"\nNeighborhood: {self.instance.neighborhood}"

        return base_prompt + state_info


# -----------------------------------------------------------------------------
# Specialized agents for different use cases
# -----------------------------------------------------------------------------


class PlaceReviewerAgent(PlaceAgent):
    """
    A specialized agent for reviewing and approving proposed changes.

    This agent focuses on validating changes before they're applied,
    useful in multi-agent workflows where one agent proposes and another approves.
    """

    base_prompt: ClassVar[str] = """
You are a careful reviewer of proposed changes to place records.

Your job is to:
1. Review proposed changes for accuracy and completeness
2. Check that URLs are well-formed and point to legitimate services
3. Verify that descriptions are appropriate and accurate
4. Approve valid changes and reject problematic ones

Be conservative - when in doubt, reject and ask for clarification.
"""

    tools: ClassVar[list[type[ModelTool]]] = [
        GetPlaceInfoTool,
        GetDeliveryOptionsTool,
        GetHoursTool,
    ]


class PlaceDataEntryAgent(PlaceAgent):
    """
    Agent optimized for bulk data entry tasks.

    Has access to update tools but not state-changing tools.
    """

    base_prompt: ClassVar[str] = """
You are a data entry assistant helping to populate place records.

Your job is to:
1. Accept new information about places
2. Update records with provided data
3. Ask clarifying questions when information is ambiguous
4. Validate data formats (phone numbers, URLs, etc.)

Focus on accuracy. Don't make assumptions about missing data.
"""

    tools: ClassVar[list[type[ModelTool]]] = [
        GetPlaceInfoTool,
        UpdateDescriptionTool,
        UpdateContactInfoTool,
        ProposeDeliveryUrlTool,
    ]
