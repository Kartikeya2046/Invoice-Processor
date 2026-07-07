"""FastAPI entry point for the invoice processor service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging

import database
from routes import search, upload, structured

logger = logging.getLogger(__name__)

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
    if database.database is None:
        print("MongoDB connection unavailable; starting FastAPI without database access")

    cuda_available = torch.cuda.is_available()
    print(f"[STARTUP] CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[STARTUP] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[STARTUP] VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    else:
        print("[STARTUP] WARNING: CUDA not available — Surya OCR will fall back to CPU")


@app.on_event("shutdown")
async def shutdown_event():
    if database.client is not None:
        database.client.close()