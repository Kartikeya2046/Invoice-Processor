from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ProcessingStatus(str, Enum):
    pending = "pending"
    extracted = "extracted"
    review_required = "review_required"
    failed = "failed"

class LineItem(BaseModel):
    line_number: Optional[str] = None
    part_number: Optional[str] = None
    manufacturer_part_number: Optional[str] = None
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
    shipping_amount: Optional[float] = None
    grand_total: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)

class ValidationResult(BaseModel):
    confidence_score: float = 0.0
    needs_review: bool = True
    missing_fields: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)

class InvoiceData(BaseModel):
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
    processing_logs: Optional[str] = None
    correction_history: List[Dict[str, Any]] = Field(default_factory=list)

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
