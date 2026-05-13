# AutoInsight Description

AutoInsight is an autonomous multi-agent system that automates the complete machine learning workflow for big data, from dataset ingestion to model training, evaluation, and report generation.

The project combines:

- Multi-agent AI orchestration using LangGraph
- FastAPI backend for job execution and APIs
- React + Vite frontend for real-time monitoring
- LLM-driven decision making for preprocessing and modeling

---

# Features

- Intelligent goal parsing from plain English
- Automatic task detection (classification/regression)
- Automated EDA and data validation
- AI-driven cleaning and feature engineering
- Multi-model training and best-model selection
- Real-time pipeline tracking with live logs
- Interactive dashboard with metrics and feature importance
- Automatic markdown report generation
- Supports CSV, Excel, JSON, Parquet, URLs, and Google Sheets

---

# Project Structure

```text
AutoInsight/ 
├── main.py 
├── pyproject.toml 
├── automl/ 
│ ├── __init__.py 
│ ├── graph.py 
│ ├── state.py
│ ├── llm.py 
│ ├── agents/ 
│ │ ├── goal_parser.py 
│ │ ├── eda.py 
│ │ ├── strategy.py 
│ │ ├── validator.py 
│ │ ├── cleaning.py 
│ │ ├── feature.py 
│ │ ├── modeling.py
│ │ └── report.py 
│ └── tools/ 
│ ├── profiling.py
│ ├── cleaning.py 
│ ├── feature_engineering.py 
│ └── modeling.py
└── requirements.txt

autoinsight-ui/ 
├── backend/ 
│ ├── main.py
│ ├── models.py
│ ├── runner.py 
│ ├── requirements.txt 
│ ├── start.bat
│ └── start.sh
└── frontend/ 
├── src/ 
│ ├── App.jsx 
│ ├── api.js
│ ├── index.css 
│ └── components/ 
│ ├── DataSourcePanel.jsx 
│ ├── StageProgress.jsx 
│ ├── LogTerminal.jsx
│ ├── ResultsDashboard.jsx
│ ├── ReportViewer.jsx 
│ └── JobHistory.jsx 
├── index.html 
├── vite.config.js 
├── package.json 
├── start.bat 
└── start.sh
```

---

## Prerequisites

- Python 3.10+ with your AutoInsight venv active
- Node.js 18+
- The AutoInsight project folder at `../../AutoInsight` relative to this folder

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

## How it works

1. **Upload** a CSV/Excel/JSON/Parquet file, or paste a URL/Google Sheets link
2. **Describe** your ML goal in plain English
3. **Choose** an LLM backend (Auto tries Groq → Gemini)
4. **Click Run** — the job starts in a background thread
5. **Watch** the live log terminal and stage tracker update in real time via SSE
6. **Review** the results dashboard (model scores, feature importance charts)
7. **Read or download** the full markdown report

Jobs persist in memory for the lifetime of the server process.

```

---

# Tech Stack

## Backend & AI
- Python 3.11+
- LangGraph
- FastAPI
- Scikit-learn
- Pandas
- Groq API

## Frontend
- React
- Vite
- Server-Sent Events (SSE)

---

# Quick Start

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd AutoInsight
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

### Backend

```bash
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

# Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Run the Project

## Start Backend

### Windows

```bash
cd backend
start.bat
```

### Mac/Linux

```bash
cd backend
chmod +x start.sh
./start.sh
```

Backend:

```text
http://localhost:8000
```

---

## Start Frontend

Open another terminal.

### Windows

```bash
cd frontend
start.bat
```

### Mac/Linux

```bash
cd frontend
chmod +x start.sh
./start.sh
```

Frontend:

```text
http://localhost:5173
```

---

# Example CLI Usage

```bash
python main.py \
  --csv data/housing.csv \
  --goal "Predict house prices based on property features" \
  --output report.md
```

---

# Generated Outputs

- Automated EDA
- Data quality analysis
- Model comparison metrics
- Feature importance charts
- Final markdown report



# Contributions are welcome.
