# Document Extraction Framework

A deterministic-first, multi-source document extraction system built with FastAPI, Celery, pdfminer, Tesseract, and PostgreSQL/MinIO.

## Architecture

```
┌─────────────┐     POST /v1/extract     ┌─────────────────┐
│   Client    │ ─────────────────────▶  │   FastAPI API   │
└─────────────┘                          └────────┬────────┘
                                                  │ Celery task
                                          ┌───────▼────────┐
                                          │  Pipeline Orch  │
                                          └───────┬────────┘
                    ┌─────────────────────────────┤
                    │                             │
           ┌────────▼──────────┐      ┌──────────▼────────┐
           │  Normalization    │      │  Field Config       │
           │  (PDF/OCR/Image)  │      │  (YAML per doctype)│
           └────────┬──────────┘      └──────────┬────────┘
                    │                             │
           ┌────────▼──────────┐                 │
           │  Deterministic    │◀────────────────┘
           │  Extractor        │
           │  (Anchor + Regex) │
           └────────┬──────────┘
                    │
           ┌────────▼──────────┐
           │  Validators +     │
           │  Confidence Score │
           └────────┬──────────┘
                    │ (if low confidence)
           ┌────────▼──────────┐
           │  LLM Fallback     │  ← optional, behind feature flag
           │  (Ollama/Cloud)   │
           └────────┬──────────┘
                    │
           ┌────────▼──────────┐
           │  Postgres + MinIO │
           └───────────────────┘
```

## Repo Structure

```
docextract/
├── app/
│   ├── api/                # FastAPI routes
│   │   └── routes.py
│   ├── config/             # YAML field configs
│   │   ├── invoice.yaml
│   │   └── loader.py
│   ├── extractors/         # Deterministic extraction
│   │   └── deterministic.py
│   ├── fallback/           # LLM fallback
│   │   ├── gate.py
│   │   └── llm_provider.py
│   ├── normalization/      # PDF text + OCR pipeline
│   │   ├── file_router.py
│   │   ├── ocr_pipeline.py
│   │   └── pdf_extractor.py
│   ├── pipeline/           # Orchestrator
│   │   └── orchestrator.py
│   ├── storage/            # Postgres + MinIO
│   │   ├── db.py
│   │   ├── minio_client.py
│   │   └── queries.py
│   ├── validators/         # Normalizers + validators
│   │   ├── normalizers.py
│   │   └── rules.py
│   ├── workers/            # Celery tasks
│   │   └── celery_app.py
│   ├── main.py             # FastAPI app entry
│   ├── models.py           # Canonical data models
│   └── settings.py         # App settings (env vars)
├── tests/
│   ├── fixtures/           # Sample PDFs (generated)
│   └── unit/               # Unit + integration tests
├── scripts/
│   └── generate_fixtures.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites
- Docker 24+ and Docker Compose v2
- 4GB RAM minimum (Tesseract + workers)

### 1. Start all services

```bash
docker compose up --build
```

This starts: FastAPI API (port 8000), Celery worker, PostgreSQL, Redis, MinIO (port 9000/9001).

Wait for all services to be healthy (~30s). You should see:
```
api_1     | {"event": "db_initialized", ...}
api_1     | {"event": "minio_ready", ...}
```

### 2. Generate test fixtures (optional)

```bash
docker compose run --rm api python scripts/generate_fixtures.py
```

Or on your host (requires `reportlab` or `fpdf2`):
```bash
pip install reportlab
python scripts/generate_fixtures.py
```

### 3. Upload a document

```bash
# Upload an invoice PDF
curl -X POST http://localhost:8000/v1/extract \
  -F "file=@tests/fixtures/invoice_digital_1.pdf" \
  -F "document_type=invoice" \
  -F "supplier_id=acme-001"
```

Response:
```json
{
  "job_id": "3f2a1b4c-...",
  "document_id": "9d8e7f6a-...",
  "message": "Extraction job submitted"
}
```

### 4. Check job status

```bash
curl http://localhost:8000/v1/jobs/3f2a1b4c-...
```

Response when completed:
```json
{
  "job_id": "3f2a1b4c-...",
  "document_id": "9d8e7f6a-...",
  "status": "completed",
  "result": {
    "fields": {
      "invoice_number": {
        "value": "INV-2024-001",
        "confidence": 0.92,
        "status": "verified",
        "method": "anchor",
        "source": {"page": 0, "bbox": {"x0": 300, "y0": 150, "x1": 420, "y1": 165}},
        "evidence": {"snippet": "Invoice Number: INV-2024-001"}
      },
      "total_amount": {
        "value": 4895.0,
        "confidence": 0.88,
        "status": "verified",
        "method": "anchor"
      }
    },
    "validation_summary": {
      "required_present": true,
      "required_missing": [],
      "overall_confidence": 0.87
    }
  }
}
```

### 5. Get latest result for a document

```bash
curl http://localhost:8000/v1/documents/9d8e7f6a-.../result
```

### 6. List artifacts

```bash
curl http://localhost:8000/v1/documents/9d8e7f6a-.../artifacts
```

## Running Tests

```bash
# Inside Docker
docker compose run --rm api pytest tests/ -v

# Locally (requires pdfminer.six, pydantic, etc.)
pip install -r requirements.txt
pip install reportlab  # for fixture generation
python scripts/generate_fixtures.py
pytest tests/ -v
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `docextract` | MinIO bucket name |
| `DEBUG` | `false` | Enable debug traces |
| `LLM_FALLBACK_ENABLED` | `false` | Enable LLM fallback |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Local Ollama URL |
| `LLM_MODEL` | `llama3` | Ollama model name |
| `LLM_CONFIDENCE_THRESHOLD` | `0.5` | Trigger LLM below this |
| `CLOUD_LLM_PROVIDER` | `` | `openai` or `anthropic` |
| `CLOUD_LLM_API_KEY` | `` | Cloud provider API key |

### Adding a New Document Type

1. Create `app/config/<doctype>.yaml` following the `invoice.yaml` schema
2. Define fields with `anchors`, `patterns`, `normalizers`, `validators`
3. No code changes needed — the config system handles the rest

Example `purchase_order.yaml`:
```yaml
document_type: purchase_order
fields:
  po_number:
    type: string
    required: true
    anchors: ["po number", "purchase order", "po #"]
    patterns: ['(?:po\s*(?:number|#))[:\s]+([\w\-]+)']
    ...
```

### Enabling LLM Fallback with Ollama

```bash
# Pull a model
docker run -it --rm ollama/ollama pull llama3

# Add Ollama service to docker-compose.yml and set:
LLM_FALLBACK_ENABLED=true
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=llama3
```

LLM is triggered only when:
- A required field is missing after deterministic extraction
- Field confidence is below `LLM_CONFIDENCE_THRESHOLD`
- Field has `fallback_allowed: true` in config

## Extraction Pipeline

### Stage 1: File Detection
- `python-magic` detects MIME type (PDF, PNG, JPEG, TIFF)
- Digital PDFs → pdfminer.six (token-level bboxes)
- Scanned PDFs / Images → rasterize (pdf2image) → Tesseract OCR

### Stage 2: Normalization
- Unified `NormalizedDocument` with per-page tokens (text, bbox, line_id)
- OCR: preprocess (grayscale → denoise → adaptive threshold → deskew) then Tesseract TSV

### Stage 3: Deterministic Extraction
- **Anchor extraction**: fuzzy-match anchor phrases (e.g., "Invoice No", "Inv #") → extract adjacent tokens
- **Regex extraction**: apply patterns over full text → score candidates
- **Scoring**: `method_weight + normalization_bonus + validation_bonus - conflict_penalty`

### Stage 4: Validation
- Type parsing (date, money, string)
- Per-field validators (min_length, positive_number, valid_date, not_future)
- Cross-field checks (total ≥ tax, due_date ≥ invoice_date)
- Status: `verified` (≥0.85), `needs_review` (0.50–0.85), `missing` (<0.50)

### Stage 5: LLM Fallback (optional)
- Triggered only for required-missing or low-confidence fields
- Sends minimal document snippet to LLM
- Enforces strict JSON response schema
- Supports Ollama (local) and cloud providers (OpenAI, Anthropic)

## Debug Mode

Set `DEBUG=true` to persist `debug_trace.json` per job in MinIO:
```json
{
  "invoice_number": {
    "method": "anchor",
    "confidence": 0.92,
    "status": "verified",
    "candidates": [
      {"value": "INV-2024-001", "confidence": 0.92, "method": "anchor"},
      {"value": "INV-2024-001", "confidence": 0.70, "method": "regex"}
    ]
  }
}
```

## MinIO Console

Access the MinIO console at http://localhost:9001 with credentials `minioadmin/minioadmin`.

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```
