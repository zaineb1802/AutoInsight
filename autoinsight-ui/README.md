# AutoInsight UI

Web interface for the AutoInsight AutoML pipeline.
FastAPI backend + Vite/React frontend.

```
autoinsight-ui/
├── backend/
│   ├── main.py          # FastAPI app + all endpoints
│   ├── models.py        # Pydantic request/response schemas
│   ├── runner.py        # Background job execution + SSE log capture
│   ├── requirements.txt
│   ├── start.bat        # Windows launcher
│   └── start.sh         # Mac/Linux launcher
└── frontend/
    ├── src/
    │   ├── App.jsx                        # Root layout + state
    │   ├── api.js                         # All fetch/SSE calls
    │   ├── index.css                      # Design system tokens
    │   └── components/
    │       ├── DataSourcePanel.jsx        # File upload / URL / GSheets
    │       ├── StageProgress.jsx          # 8-stage pipeline tracker
    │       ├── LogTerminal.jsx            # Live SSE log stream
    │       ├── ResultsDashboard.jsx       # Model scores + feature importance charts
    │       ├── ReportViewer.jsx           # Rendered markdown report
    │       └── JobHistory.jsx             # Sidebar run history
    ├── index.html
    ├── vite.config.js   # Dev server + /api proxy to :8000
    ├── package.json
    ├── start.bat
    └── start.sh
```

---

## Prerequisites

- Python 3.10+ with your AutoInsight venv active
- Node.js 18+
- The AutoInsight project folder at `../../AutoInsight` relative to this folder
  (i.e. sibling directory), OR set `PYTHONPATH` manually

Recommended folder layout:
```
Desktop/
├── AutoInsight/      ← existing project
└── autoinsight-ui/   ← this folder
```

---

## Setup & Run

### 1. Start the backend  (Terminal 1)

**Windows:**
```bat
cd autoinsight-ui\backend
start.bat
```

**Mac/Linux:**
```bash
cd autoinsight-ui/backend
chmod +x start.sh && ./start.sh
```

The API starts on **http://localhost:8000**
Swagger docs at **http://localhost:8000/docs**

### 2. Start the frontend  (Terminal 2)

**Windows:**
```bat
cd autoinsight-ui\frontend
start.bat
```

**Mac/Linux:**
```bash
cd autoinsight-ui/frontend
chmod +x start.sh && ./start.sh
```

Frontend at **http://localhost:5173**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload a dataset file |
| `POST` | `/api/run` | Start an AutoML job |
| `GET`  | `/api/jobs/{id}` | Poll job status + results |
| `GET`  | `/api/jobs/{id}/logs` | SSE stream of live log lines |
| `GET`  | `/api/jobs/{id}/report` | Fetch final markdown report |
| `GET`  | `/api/jobs` | List all jobs |
| `DELETE` | `/api/jobs/{id}` | Remove a job record |

---

## How it works

1. **Upload** a CSV/Excel/JSON/Parquet file, or paste a URL/Google Sheets link
2. **Describe** your ML goal in plain English
3. **Choose** an LLM backend (Auto tries Groq → Gemini)
4. **Click Run** — the job starts in a background thread
5. **Watch** the live log terminal and stage tracker update in real time via SSE
6. **Review** the results dashboard (model scores, feature importance charts)
7. **Read or download** the full markdown report

Jobs persist in memory for the lifetime of the server process.
