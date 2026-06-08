"""Pydantic models used to validate invoice and search data."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
	pending = "pending"
	processing = "processing"
	extracted = "extracted"
	review_required = "review_required"
	reviewed = "reviewed"
	failed = "failed"


class LineItem(BaseModel):
	description: Optional[str] = None
	quantity: Optional[float] = None
	unit_price: Optional[float] = None
	total_price: Optional[float] = None


class ExtractedInvoiceData(BaseModel):
	invoice_number: Optional[str] = None
	invoice_date: Optional[str] = None
	due_date: Optional[str] = None
	vendor_name: Optional[str] = None
	vendor_address: Optional[str] = None
	vendor_email: Optional[str] = None
	vendor_phone: Optional[str] = None
	customer_name: Optional[str] = None
	customer_address: Optional[str] = None
	billing_address: Optional[str] = None
	shipping_address: Optional[str] = None
	po_number: Optional[str] = None
	payment_terms: Optional[str] = None
	currency: Optional[str] = None
	subtotal: Optional[float] = None
	tax_amount: Optional[float] = None
	tax_rate: Optional[float] = None
	discount_amount: Optional[float] = None
	grand_total: Optional[float] = None
	notes: Optional[str] = None
	line_items: List[LineItem] = Field(default_factory=list)


class ValidationResult(BaseModel):
	is_valid: bool = False
	confidence_score: float = 0.0
	missing_fields: List[str] = Field(default_factory=list)
	failed_checks: List[str] = Field(default_factory=list)
	needs_review: bool = True


class CorrectionRecord(BaseModel):
	field_name: str
	original_value: Any = None
	corrected_value: Any = None
	corrected_at: datetime = Field(default_factory=datetime.utcnow)
	corrected_by: str = "user"


class InvoiceData(BaseModel):
	"""Represents extracted invoice data stored in MongoDB."""

	filename: str
	extracted_text: str
	metadata: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	processing_status: ProcessingStatus = ProcessingStatus.pending
	document_type: Optional[str] = None
	document_type_confidence: Optional[float] = None
	extracted_fields: Optional[ExtractedInvoiceData] = None
	validation_result: Optional[ValidationResult] = None
	extraction_error: Optional[str] = None
	correction_history: List[CorrectionRecord] = Field(default_factory=list)


class SearchQuery(BaseModel):
	"""Represents the search input submitted to the invoice search API."""

	query: str
	limit: int = 10

