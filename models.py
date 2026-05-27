"""Pydantic models used to validate invoice and search data."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class InvoiceData(BaseModel):
	"""Represents extracted invoice data stored in MongoDB."""

	filename: str
	extracted_text: str
	metadata: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
	"""Represents the search input submitted to the invoice search API."""

	query: str
	limit: int = 10

