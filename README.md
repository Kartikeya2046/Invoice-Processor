# 🧾 Invoice Processor

> An intelligent OCR-powered invoice scanning and search system — upload invoices as PDFs or images, extract their text automatically, and search across all stored invoices in seconds.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-invoice--processor--pink.vercel.app-blue)](https://invoice-processor-pink.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)](https://mongodb.com)

---

## 📌 Overview

Invoice Processor automates the tedious work of digitizing paper and scanned invoices. It accepts PDF and image uploads, runs OCR (via Tesseract) to extract text, stores the results in MongoDB, and exposes a full-text search API so you can instantly find any invoice by vendor, amount, date, or any keyword.

---

## ✨ Features

- **Multi-format upload** — supports PDF, JPG, JPEG, PNG, BMP, TIFF
- **Smart OCR pipeline** — native PDF text extraction with automatic fallback to Tesseract for scanned documents
- **Full-text search** — MongoDB text index enables fast keyword search across all stored invoices
- **REST API** — clean FastAPI endpoints with automatic Swagger docs at `/docs`
- **Cloud-ready** — ships with `Procfile`, `railway.json`, and `vercel.json` for zero-config deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Frontend (HTML/JS)              │
│         invoice-frontend.html / index.html   │
└──────────────────┬──────────────────────────┘
                   │ HTTP (REST)
┌──────────────────▼──────────────────────────┐
│            FastAPI Backend (main.py)         │
│  ┌────────────────┐  ┌─────────────────────┐│
│  │  /upload route │  │   /search route     ││
│  └───────┬────────┘  └─────────┬───────────┘│
│          │                     │            │
│  ┌───────▼────────┐   ┌────────▼───────────┐│
│  │ ocr_processor  │   │    database.py      ││
│  │ (Tesseract +   │   │  (MongoDB + text    ││
│  │  pdfplumber)   │   │    index search)    ││
│  └───────┬────────┘   └────────────────────┘│
└──────────┼──────────────────────────────────┘
           │ stores extracted text
┌──────────▼──────────────────────────────────┐
│              MongoDB Atlas                   │
│           Collection: invoices               │
└─────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system
- MongoDB Atlas account (or local MongoDB)
- Poppler (required by `pdf2image` for scanned PDF conversion)

### 1. Clone the repository

```bash
git clone https://github.com/Kartikeya2046/Invoice-Processor.git
cd Invoice-Processor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `MONGO_URI` | MongoDB connection string |
| `MONGO_DB_NAME` | Database name (e.g. `invoice_db`) |
| `TESSERACT_PATH` | Absolute path to `tesseract` binary |

### 4. Run the server

```bash
uvicorn main:app --reload
```

The API will be live at `http://localhost:8001`. Visit `http://localhost:8001/docs` for the interactive Swagger UI.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload and process an invoice file |
| `POST` | `/search` | Search invoices by keyword |

### Upload an invoice

```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@invoice.pdf"
```

**Response:**
```json
{
  "message": "Invoice processed successfully",
  "filename": "invoice.pdf",
  "extracted_text": "INVOICE #1234 ..."
}
```

### Search invoices

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Acme Corp", "limit": 10}'
```

---

## 📂 Project Structure

```
Invoice-Processor/
├── main.py               # FastAPI app entry point, CORS, router wiring
├── ocr_processor.py      # OCR logic: PDF extraction + Tesseract fallback
├── database.py           # MongoDB connection and collection setup
├── models.py             # Pydantic data models (InvoiceData, SearchQuery)
├── config.py             # Environment variable loading
├── routes/
│   ├── upload.py         # POST /upload — file ingestion and OCR
│   └── search.py         # POST /search — full-text search
├── index.html            # Landing page
├── invoice-frontend.html # Main upload/search UI
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── Procfile              # Heroku/Railway process definition
├── railway.json          # Railway deployment config
└── vercel.json           # Vercel deployment config
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI + Uvicorn |
| **OCR Engine** | Tesseract (via pytesseract) |
| **PDF Parsing** | pdfplumber (native), pdf2image + Tesseract (scanned) |
| **Image Processing** | Pillow |
| **Database** | MongoDB Atlas (pymongo) |
| **Data Validation** | Pydantic v2 |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **Deployment** | Vercel (frontend) / Railway (backend) |

---

## ☁️ Deployment

### Railway (Backend)

The `Procfile` and `railway.json` are pre-configured. Connect your GitHub repo to Railway, add your environment variables, and deploy.

### Vercel (Frontend)

The `vercel.json` routes all requests through the FastAPI app. Set the same environment variables in your Vercel project settings.
