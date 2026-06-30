from datetime import datetime
from fastapi import APIRouter, HTTPException, Body
from bson import ObjectId
from database import invoices_collection

router = APIRouter()


@router.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    if not ObjectId.is_valid(invoice_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")

    doc["_id"] = str(doc["_id"])
    return doc


@router.patch("/invoices/{invoice_id}/fields")
def update_invoice_field(invoice_id: str, payload: dict = Body(...)):
    if not ObjectId.is_valid(invoice_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Invoice not found")

    field_name = payload.get("field_name")
    if "corrected_value" in payload:
        new_value = payload.get("corrected_value")
    else:
        new_value = payload.get("value")

    if not field_name:
        raise HTTPException(status_code=400, detail="Missing field_name")

    # BOE documents store their fields under extracted_boe_fields, not
    # extracted_fields -- pick the matching key for this document's type
    # rather than guessing one and silently writing to the wrong place.
    fields_key = "extracted_boe_fields" if doc.get("document_type") == "bill_of_entry" else "extracted_fields"
    extracted = doc.get(fields_key) or {}
    old_value = extracted.get(field_name)

    invoices_collection.update_one(
        {"_id": ObjectId(invoice_id)},
        {
            "$set": {f"{fields_key}.{field_name}": new_value},
            "$push": {"correction_history": {
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": datetime.utcnow(),
            }},
        },
    )

    updated_doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    updated_doc["_id"] = str(updated_doc["_id"])
    return updated_doc


@router.post("/invoices/{invoice_id}/reprocess")
def reprocess_invoice(invoice_id: str):
    if not ObjectId.is_valid(invoice_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")
    return {"status": "reprocessed"}


@router.patch("/invoices/{invoice_id}/mark-reviewed")
def mark_reviewed(invoice_id: str):
    if not ObjectId.is_valid(invoice_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")
    invoices_collection.update_one(
        {"_id": ObjectId(invoice_id)},
        {"$set": {"processing_status": "extracted"}}
    )
    return {"status": "marked as reviewed"}