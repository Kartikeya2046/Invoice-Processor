"""Search and listing endpoints for stored invoices."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from database import invoices_collection
from models import SearchQuery


router = APIRouter()


def _serialize_invoice(document: Dict[str, Any]) -> Dict[str, Any]:
	if "_id" in document:
		document["_id"] = str(document["_id"])
	return document


@router.post("/search")
async def search_invoices(payload: SearchQuery):
	try:
		cursor = (
			invoices_collection.find(
				{"$text": {"$search": payload.query}},
				{"score": {"$meta": "textScore"}},
			)
			.sort([("score", {"$meta": "textScore"})])
			.limit(payload.limit)
		)

		results = [_serialize_invoice(doc) for doc in cursor]
		return {
			"query": payload.query,
			"count": len(results),
			"results": results,
		}
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Search failed: {error}")


@router.get("/invoices")
async def list_invoices(limit: int = 50, skip: int = 0):
	if limit < 1 or skip < 0:
		raise HTTPException(status_code=400, detail="Invalid pagination parameters")

	try:
		total = invoices_collection.count_documents({})
		cursor = (
			invoices_collection.find()
			.sort("created_at", -1)
			.skip(skip)
			.limit(limit)
		)

		invoices = [_serialize_invoice(doc) for doc in cursor]
		return {
			"total": total,
			"count": len(invoices),
			"invoices": invoices,
		}
	except Exception as error:
		raise HTTPException(status_code=500, detail=f"Failed to list invoices: {error}")


@router.get("/health")
async def health_check():
	return {"status": "healthy"}

