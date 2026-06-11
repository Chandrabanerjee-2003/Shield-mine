# ⚒️ Mining Accident Analytics & RAG System

A Streamlit-based data analytics and Retrieval-Augmented Generation (RAG) system for exploring, visualizing, and querying mining accident datasets. Built to support both Indian and Australian mining accident records, with schema-agnostic design that works with any country's dataset.

---

## Features

**Universal Dataset Support**
- Works with Excel (`.xlsx`, `.xls`) and CSV files
- Auto-detects column roles — location, fatality count, date, owner, accident category — regardless of column naming conventions
- Displays detected column mapping on upload for full transparency
- Multi-file upload: analyze and compare datasets side by side

**Data Analysis & Visualization**
- Interactive Plotly charts: yearly trends, category breakdowns, state/region rankings
- Risk score computation per accident category (frequency × severity)
- Anomaly detection using Z-score on yearly incident counts
- Chi-squared test heatmap for categorical correlations
- K-means clustering with PCA visualization
- Word cloud generation from incident descriptions and safety suggestions
- Age, gender, job role, and shift-wise analysis (auto-detected from dataset columns)

**Hybrid RAG System**
- BM25 (keyword search) + FAISS (semantic vector search) ensemble retrieval
- Cross-Encoder reranking for precision
- ChatGroq (LLaMA) as the LLM backend for natural language answers
- Source document transparency — see exactly which records were used to generate each answer
- Smart rebuild: full rebuild when embedding/reranker model changes, fast update when only the chat model changes
- Persistent chat history within a session

**Safety Audit Report Generator**
- Auto-generates a structured safety audit report with KPIs, fatality trends, geographic analysis, and root cause summary
- AI-written safety recommendations powered by Groq LLM
- Download as Markdown or PDF

---

## Project Structure

```
├── combined.py          # Main Streamlit application
├── analytics_engine.py  # Core analytics and visualization functions
├── .env                 # API keys (not committed)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/mining-accident-analytics.git
cd mining-accident-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free Groq API key at [console.groq.com](https://console.groq.com).

**4. Run the app**
```bash
streamlit run combined.py
```

---

## Models Used

| Component | Option 1 | Option 2 |
|-----------|----------|----------|
| Embedding | `all-MiniLM-L6-v2` (fast) | `all-mpnet-base-v2` (higher quality) |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `ms-marco-MiniLM-L-12-v2` (best quality) |
| Chat LLM | `llama-3.1-8b-instant` (fast) | `llama-3.1-70b-versatile` (best quality) |

---

## Sample Datasets

Two datasets are bundled as downloadable links inside the app (in the **Get Datasets** tab):

| Dataset | Records | Period |
|---------|---------|--------|
| 🇮🇳 India Mining Accidents | 337 | 2016 – 2022 |
| 🇦🇺 Australia Mining Accidents | 350 | 1882 – 2024 |

---

## Tech Stack

- **Frontend:** Streamlit
- **Analytics:** Pandas, NumPy, Scikit-learn, SciPy
- **Visualization:** Plotly, Seaborn, Matplotlib, WordCloud
- **RAG Pipeline:** LangChain, FAISS, BM25Retriever, CrossEncoder (sentence-transformers)
- **LLM:** ChatGroq (LLaMA 3.1 via Groq API)
- **Embeddings:** HuggingFace sentence-transformers


---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
