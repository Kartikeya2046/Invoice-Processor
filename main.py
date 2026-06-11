"""FastAPI entry point for the invoice processor service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
from routes import search, upload, structured



app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(structured.router)


@app.get("/health")
def health_check():
	return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
	import os
	mistral_key = os.getenv("MISTRAL_API_KEY", "NOT SET")
	mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
	print(f"MISTRAL_API_KEY: {'SET (' + mistral_key[:8] + '...)' if mistral_key != 'NOT SET' else 'NOT SET — extraction will fail'}")
	print(f"MISTRAL_MODEL: '{mistral_model}'")
	if database.database is None:
		print("MongoDB connection unavailable; starting FastAPI without database access")


@app.on_event("shutdown")
async def shutdown_event():
	if database.client is not None:
		database.client.close()

