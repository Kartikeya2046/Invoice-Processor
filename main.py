"""FastAPI entry point for the invoice processor service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
from routes import search, upload
from routes import structured



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
from routes import structured
app.include_router(structured.router)


@app.get("/health")
def health_check():
	return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
	import os
	groq_key = os.getenv("GROQ_API_KEY", "NOT SET")
	print(f"GROQ_API_KEY status: {'SET ('+groq_key[:8]+'...)' if groq_key != 'NOT SET' else 'NOT SET - extraction will fail'}")
	if database.database is None:
		print("MongoDB connection unavailable; starting FastAPI without database access")


@app.on_event("shutdown")
async def shutdown_event():
	if database.client is not None:
		database.client.close()

