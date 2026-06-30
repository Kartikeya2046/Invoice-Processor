"""Pydantic models for invoice / Bill of Entry structured extraction.

Field set follows the actual spec (PO Number, Supplier, Invoice Number,
Invoice Date, Quantity, Unit Price, CGST, SGST for invoices; BOE Number,
BOE Date, IGST, Cust. Duty, SBCESS for Bill of Entry). CGST/SGST are
populated only when present in the document -- their presence/absence is
what distinguishes a Local invoice from the invoice side of a Cargo/Import
shipment, so there's no separate "is_import" flag to maintain.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    pending = "pending"
    extracted = "extracted"
    review_required = "review_required"
    failed = "failed"


class LineItem(BaseModel):
    """A single product row inside an invoice's line items table."""

    line_number: Optional[str] = None
    product_id: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None


class ExtractedInvoiceData(BaseModel):
    """Fields extracted from an Invoice (Cargo/Import side, or Local)."""

    po_number: Optional[str] = None
    supplier: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    quantity: Optional[str] = None  # header/total quantity, distinct from per-line quantity
    unit_price: Optional[str] = None  # header/total unit price, distinct from per-line unit_price
    cgst: Optional[str] = None  # Local invoices only; null on Cargo/Import invoices
    sgst: Optional[str] = None  # Local invoices only; null on Cargo/Import invoices
    line_items: List[LineItem] = Field(default_factory=list)


class ExtractedBOEData(BaseModel):
    """Fields extracted from a Bill of Entry (Cargo/Import only)."""

    boe_number: Optional[str] = None
    boe_date: Optional[str] = None
    igst: Optional[str] = None
    cust_duty: Optional[str] = None
    sbcess: Optional[str] = None


class ValidationResult(BaseModel):
    """Outcome of validating extracted fields before storage."""

    confidence_score: float = 0.0
    needs_review: bool = True
    missing_fields: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class CorrectionRecord(BaseModel):
    """One manual field correction, for audit history."""

    field: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InvoiceData(BaseModel):
    """A processed document and everything the pipeline produced for it."""

    filename: str
    extracted_text: str = ""
    processing_status: ProcessingStatus = ProcessingStatus.pending
    document_type: Optional[str] = None  # "invoice" or "bill_of_entry"
    document_type_confidence: Optional[float] = None
    extracted_fields: Optional[ExtractedInvoiceData] = None
    extracted_boe_fields: Optional[ExtractedBOEData] = None
    validation_result: Optional[ValidationResult] = None
    extraction_error: Optional[str] = None
    correction_history: List[CorrectionRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search input submitted to the invoice search API."""

    query: str
    limit: int = 10
