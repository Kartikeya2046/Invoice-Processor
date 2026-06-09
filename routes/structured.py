"""Structured data endpoints for invoice intelligence."""

from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import invoices_collection
from models import ProcessingStatus, ValidationResult, CorrectionRecord
from validator import validate_extracted_fields

router = APIRouter()


def _serialize(doc: dict) -> dict:
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class FieldCorrection(BaseModel):
    field_name: str
    corrected_value: object


@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    try:
        doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID")
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _serialize(doc)


@router.get("/invoices/{invoice_id}/structured")
async def get_structured(invoice_id: str):
    try:
        doc = invoices_collection.find_one(
            {"_id": ObjectId(invoice_id)},
            {"extracted_fields": 1, "validation_result": 1, "processing_status": 1}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID")
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _serialize(doc)


@router.patch("/invoices/{invoice_id}/fields")
async def correct_field(invoice_id: str, correction: FieldCorrection):
    try:
        doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID")
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")

    extracted = doc.get("extracted_fields") or {}
    original_value = extracted.get(correction.field_name)

    correction_record = {
        "field_name": correction.field_name,
        "original_value": original_value,
        "corrected_value": correction.corrected_value,
        "corrected_at": datetime.utcnow().isoformat(),
        "corrected_by": "user",
    }

    update_payload = {
        f"extracted_fields.{correction.field_name}": correction.corrected_value
    }

    invoices_collection.update_one(
        {"_id": ObjectId(invoice_id)},
        {
            "$set": update_payload,
            "$push": {"correction_history": correction_record},
        }
    )

    updated_doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    updated_extracted = updated_doc.get("extracted_fields") or {}
    new_validation = validate_extracted_fields(updated_extracted)
    invoices_collection.update_one(
        {"_id": ObjectId(invoice_id)},
        {"$set": {"validation_result": new_validation}}
    )

    return {"message": "Field corrected", "field": correction.field_name, "new_value": correction.corrected_value}


@router.post("/invoices/{invoice_id}/reprocess")
async def reprocess_invoice(invoice_id: str):
    from preprocessor import clean_ocr_text, classify_document
    from llm_extractor import extract_invoice_fields
    from models import ExtractedInvoiceData, LineItem

    try:
        doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID")
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")

    raw_text = doc.get("extracted_text", "")
    cleaned = clean_ocr_text(raw_text)

    try:
        raw_fields = extract_invoice_fields(cleaned)
        if not raw_fields:
            raise ValueError("LLM returned empty response")

        new_validation = validate_extracted_fields(raw_fields)
        new_status = (
            ProcessingStatus.review_required
            if new_validation["needs_review"]
            else ProcessingStatus.extracted
        )

        invoices_collection.update_one(
            {"_id": ObjectId(invoice_id)},
            {
                "$set": {
                    "extracted_fields": raw_fields,
                    "validation_result": new_validation,
                    "processing_status": new_status.value,
                    "extraction_error": None,
                    "processing_logs": None,
                },
                "$push": {
                    "correction_history": {
                        "field_name": "__reprocess__",
                        "original_value": None,
                        "corrected_value": None,
                        "corrected_at": datetime.utcnow().isoformat(),
                        "corrected_by": "system",
                    }
                }
            }
        )
        return {"message": "Reprocessed successfully", "processing_status": new_status.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {e}")


@router.get("/invoices/queue/review")
async def review_queue(limit: int = 20, skip: int = 0):
    cursor = (
        invoices_collection.find({"processing_status": "review_required"})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    total = invoices_collection.count_documents({"processing_status": "review_required"})
    return {
        "total": total,
        "invoices": [_serialize(doc) for doc in cursor]
    }


@router.patch("/invoices/{invoice_id}/mark-reviewed")
async def mark_reviewed(invoice_id: str):
    try:
        result = invoices_collection.update_one(
            {"_id": ObjectId(invoice_id)},
            {"$set": {
                "processing_status": ProcessingStatus.reviewed.value,
                "reviewed_at": datetime.utcnow().isoformat(),
            }}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID")
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return {"message": "Marked as reviewed"}


@router.get("/analytics/summary")
async def analytics_summary():
    total = invoices_collection.count_documents({})
    by_status = {}
    for status in ProcessingStatus:
        by_status[status.value] = invoices_collection.count_documents(
            {"processing_status": status.value}
        )
    pipeline = [
        {"$match": {"validation_result.confidence_score": {"$exists": True}}},
        {"$group": {"_id": None, "avg": {"$avg": "$validation_result.confidence_score"}}}
    ]
    avg_result = list(invoices_collection.aggregate(pipeline))
    avg_confidence = round(avg_result[0]["avg"], 2) if avg_result else None

    return {
        "total_invoices": total,
        "by_status": by_status,
        "average_confidence_score": avg_confidence,
        "needs_review_count": by_status.get("review_required", 0),
    }
    