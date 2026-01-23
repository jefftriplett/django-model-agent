"""
Test models for django-model-agent tests.

These models mirror the structure used in the original tests.
"""

from django.db import models


class Place(models.Model):
    """Test Place model for testing ModelAgent functionality."""

    # State choices
    STATE_DRAFT = "draft"
    STATE_PUBLIC = "public"
    STATE_FEATURED = "featured"
    STATE_CLOSED = "closed"
    STATE_TEMP_CLOSED = "temp_closed"

    STATE_CHOICES = [
        (STATE_DRAFT, "Draft"),
        (STATE_PUBLIC, "Public"),
        (STATE_FEATURED, "Featured"),
        (STATE_CLOSED, "Closed"),
        (STATE_TEMP_CLOSED, "Temporarily Closed"),
    ]

    # Basic info
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True, default="")

    # Location
    address = models.CharField(max_length=255, blank=True, default="")
    locality = models.CharField(max_length=100, blank=True, default="")
    region = models.CharField(max_length=50, blank=True, default="")

    # Contact
    phone = models.CharField(max_length=50, blank=True, default="")
    website = models.URLField(blank=True, default="")

    # State
    state = models.CharField(
        max_length=20,
        choices=STATE_CHOICES,
        default=STATE_DRAFT,
    )

    # Service options
    delivery = models.BooleanField(default=False)
    takeout = models.BooleanField(default=False)
    dinein = models.BooleanField(default=False)
    curbside = models.BooleanField(default=False)

    # Delivery URLs
    doordash_url = models.URLField(blank=True, default="")
    grubhub_url = models.URLField(blank=True, default="")
    ubereats_url = models.URLField(blank=True, default="")
    postmates_url = models.URLField(blank=True, default="")
    seamless_url = models.URLField(blank=True, default="")
    chownow_url = models.URLField(blank=True, default="")
    eatstreet_url = models.URLField(blank=True, default="")
    menufy_url = models.URLField(blank=True, default="")
    delivery_url = models.URLField(blank=True, default="")

    # Neighborhood (optional, for location grouping)
    neighborhood = models.CharField(max_length=100, blank=True, default="")

    # Admin notes
    notes = models.TextField(blank=True, default="")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "tests"

    def __str__(self):
        return self.name

    # Simple state transition methods (mimicking django-fsm)
    def publish(self):
        """Transition from draft to public."""
        if self.state != self.STATE_DRAFT:
            raise ValueError(f"Cannot publish from state: {self.state}")
        self.state = self.STATE_PUBLIC

    def feature(self):
        """Transition from public to featured."""
        if self.state != self.STATE_PUBLIC:
            raise ValueError(f"Cannot feature from state: {self.state}")
        self.state = self.STATE_FEATURED

    def unfeature(self):
        """Transition from featured to public."""
        if self.state != self.STATE_FEATURED:
            raise ValueError(f"Cannot unfeature from state: {self.state}")
        self.state = self.STATE_PUBLIC

    def close(self):
        """Transition to closed."""
        if self.state in (self.STATE_CLOSED, self.STATE_DRAFT):
            raise ValueError(f"Cannot close from state: {self.state}")
        self.state = self.STATE_CLOSED

    def temp_close(self):
        """Transition to temporarily closed."""
        if self.state not in (self.STATE_PUBLIC, self.STATE_FEATURED):
            raise ValueError(f"Cannot temp_close from state: {self.state}")
        self.state = self.STATE_TEMP_CLOSED

    def reopen(self):
        """Transition from temp_closed to public."""
        if self.state != self.STATE_TEMP_CLOSED:
            raise ValueError(f"Cannot reopen from state: {self.state}")
        self.state = self.STATE_PUBLIC

    def set_draft(self):
        """Transition back to draft."""
        self.state = self.STATE_DRAFT
