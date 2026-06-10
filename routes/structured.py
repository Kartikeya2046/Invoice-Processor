from datetime import datetime
from fastapi import APIRouter, HTTPException, Body
from bson import ObjectId
from database import invoices_collection

router = APIRouter()

@router.get("/invoices")
def get_invoices():
    cursor = invoices_collection.find().sort("created_at", -1).limit(50)
    invoices = []
    for doc in cursor:
        extracted = doc.get("extracted_fields") or {}
        validation = doc.get("validation_result") or {}
        invoices.append({
            "id": str(doc["_id"]),
            "_id": str(doc["_id"]),
            "filename": doc.get("filename"),
            "processing_status": doc.get("processing_status"),
            "confidence_score": validation.get("confidence_score", 0.0),
            "created_at": doc.get("created_at"),
            "invoice_number": extracted.get("invoice_number"),
            "vendor_name": extracted.get("vendor_name"),
            "grand_total": extracted.get("grand_total")
        })
    return {"invoices": invoices}

@router.get("/invoices/queue/review")
def get_review_queue():
    cursor = invoices_collection.find({"processing_status": "review_required"}).sort("created_at", -1).limit(50)
    invoices = []
    for doc in cursor:
        extracted = doc.get("extracted_fields") or {}
        validation = doc.get("validation_result") or {}
        invoices.append({
            "id": str(doc["_id"]),
            "_id": str(doc["_id"]),
            "filename": doc.get("filename"),
            "processing_status": doc.get("processing_status"),
            "confidence_score": validation.get("confidence_score", 0.0),
            "created_at": doc.get("created_at"),
            "invoice_number": extracted.get("invoice_number"),
            "vendor_name": extracted.get("vendor_name"),
            "grand_total": extracted.get("grand_total")
        })
    return {"invoices": invoices}

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

    extracted_fields = doc.get("extracted_fields") or {}
    old_value = extracted_fields.get(field_name)
    
    # Do manual insertion for correction history to DB
    invoices_collection.update_one(
        {"_id": ObjectId(invoice_id)},
        {
            "$set": {f"extracted_fields.{field_name}": new_value},
            "$push": {"correction_history": {
                "field": field_name,
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": datetime.utcnow()
            }}
        }
    )
    
    updated_doc = invoices_collection.find_one({"_id": ObjectId(invoice_id)})
    updated_doc["_id"] = str(updated_doc["_id"])
    return updated_doc

@router.post("/invoices/{invoice_id}/reprocess")
def reprocess_invoice(invoice_id: str):
    # Dummy implementation for frontend button
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