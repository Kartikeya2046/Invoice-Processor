"""FastAPI entry point for the invoice processor service."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
from routes import search, upload


app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		os.getenv("FRONTEND_URL", "*")
	],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(search.router)


@app.get("/health")
def health_check():
	return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
	if database.database is None:
		raise RuntimeError("MongoDB connection failed during startup")


@app.on_event("shutdown")
async def shutdown_event():
	if database.client is not None:
		database.client.close()

