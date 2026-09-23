# SIF Precursor Analytics: Workflow & Architecture Documentation

## 1. Executive Summary & Objective

The **Serious Injury and Fatality (SIF) Precursor Analytics Prototype** is an end-to-end AI/NLP system designed to convert unstructured, free-text safety incident and observation reports into structured, actionable safety intelligence. 

Industrial frontline personnel regularly submit thousands of reports categorized as *Unsafe Acts*, *Unsafe Conditions*, *Near Misses*, or *Incidents*. Traditional manual review struggles to scale, resulting in delayed recognition of systemic risk factors. This system automatically:
1. Determines whether an observation exhibits **SIF Potential** and calculates a calibrated probability.
2. Maps observations against enterprise **Life-Saving Rules (LSR)** (e.g., *Work at Height*, *Energy Isolation*, *Confined Space*).
3. Extracts core precursor entities (**Activities**, **Equipment**, **Hazards**, **Barrier Failures**, and **Consequences**).
4. Discovers complex co-occurring risk combinations across multiple operational sites using **Frequent Itemset Mining (FP-Growth)**.
5. Surfaces emerging risk spikes and site-level risk metrics via a **FastAPI backend** and an interactive **Streamlit dashboard**.

---

## 2. Technology Stack

| Domain | Technology / Library | Purpose & Implementation Details |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core language across pipelines, APIs, and frontend. |
| **NLP & Deep Learning** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Generates 384-dimensional dense semantic vector embeddings from free-text descriptions. |
| **Machine Learning** | `scikit-learn`, `numpy`, `pandas` | - Logistic Regression with tuned decision thresholds for binary SIF detection.<br>- One-vs-Rest (OvR) multi-class Logistic Regression for Life-Saving Rules (LSR).<br>- Calibrated probability prediction and model serialization (`pickle`/`joblib`). |
| **Information Extraction** | Custom Hybrid Rule + Taxonomy Engine | Lexical and regex pattern matching combined with domain taxonomy normalization (`src/extraction/normalization.py`). Extracts equipment, activities, hazards, and barrier failures. |
| **Data Mining** | `mlxtend` (FP-Growth) | Unsupervised Frequent Itemset Mining (FIM) to extract recurring precursor combinations across locations without combinatorial explosion. |
| **Backend REST API** | `FastAPI`, `uvicorn`, `pydantic` | Asynchronous REST API providing real-time inference, batch scoring, CSV upload processing, and analytical KPI endpoints. |
| **Frontend UI** | `Streamlit`, `plotly` | High-density HSE operations dashboard with interactive Plotly visualisations, KPI cards, and site risk heatmaps. |
| **Data Persistence** | CSV, JSON, SQLite (`sqlite3`) | - Input raw reports (`data/all_reports.csv`)<br>- Enriched datasets (`data/analytics/enriched_reports.csv`)<br>- Pre-computed analytics JSON artifacts (`artifacts/analytics/*.json`)<br>- Relational observation storage (`data/sif_database.sqlite`) |

---

## 3. End-to-End Data Flow Pipeline

The end-to-end data pipeline is structured into six discrete phases, beginning with the raw input text file:

```
[Raw Input File / Free-Text]
          │
          ▼
[Phase 1: Ingestion & Text Normalization]
          │
          ├──────────────────────────────────────────────┐
          ▼                                              ▼
[Phase 2A: Dense Embedding & ML Models]   [Phase 2B: NLP Precursor Extraction]
  • all-MiniLM-L6-v2 Embeddings (384-d)     • Regex & Keyword Pattern Matchers
  • SIF Logistic Regression Classifier      • Domain Taxonomy Lookup
  • Life-Saving Rule (LSR) Classifier       • Normalization & Canonicalization
          │                                              │
          └──────────────────────┬───────────────────────┘
                                 ▼
              [Phase 3: Structured Safety Intelligence]
                     (Enriched Record / CSV)
                                 │
                                 ▼
              [Phase 4: Advanced Analytics & Pattern Mining]
                • Frequent Itemset Mining (FP-Growth)
                • Multi-Criteria Risk Scoring
                • Site Density & Wilson Interval Filtering
                • Temporal Trend & Spike Detection
                                 │
                                 ▼
              [Phase 5: Storage & Analytics Caching]
                • JSON Artifacts (artifacts/analytics/*.json)
                • SQLite Database (data/sif_database.sqlite)
                                 │
                                 ▼
              [Phase 6: Consumption & Delivery Layer]
                • FastAPI Endpoints (Swagger /docs)
                • Streamlit Operational Web Dashboard
```

---

## 4. Mermaid Flowchart

Below is a detailed Mermaid flowchart illustrating the exact data transformations from the initial input text file to the user-facing interface:

```mermaid
flowchart TD
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef prepStyle fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#311b92;
    classDef modelStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef enrichStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef analyticsStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;
    classDef storageStyle fill:#f5f5f5,stroke:#616161,stroke-width:2px,color:#212121;
    classDef apiStyle fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#004d40;
    classDef uiStyle fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17;

    %% Input Subsystem
    subgraph S1["1. Input Sources"]
        IN_CSV["Raw Input CSV File<br/>(data/all_reports.csv / User Upload)"]:::inputStyle
        IN_JSON["Demo JSON Records<br/>(data/demo/demo_reports.json)"]:::inputStyle
        IN_UI["Single Free-Text Input<br/>('Analyse New Report' in UI / API)"]:::inputStyle
    end

    %% Preprocessing Subsystem
    subgraph S2["2. Ingestion & Preprocessing"]
        INGEST["Data Loader & Schema Validator<br/>(Validates report_id, site, description)"]:::prepStyle
        CLEAN["Text Preprocessor<br/>(src/preprocessing/text_preprocessor.py)<br/>• Whitespace stripping<br/>• Normalization & cleaning"]:::prepStyle
    end

    IN_CSV --> INGEST
    IN_JSON --> INGEST
    IN_UI --> CLEAN
    INGEST --> CLEAN

    %% Dual Track Model & Extraction Subsystem
    subgraph S3["3. Dual-Track Analysis Pipeline (src/pipeline/inference.py)"]
        subgraph TrackA["Track A: Semantic Classification"]
            EMBED["SentenceTransformer Embedder<br/>(all-MiniLM-L6-v2)<br/>→ 384-dimensional Dense Vector"]:::modelStyle
            SIF_CLF["SIF Classifier<br/>(Logistic Regression)<br/>• Output: 0/1 SIF Potential<br/>• Calibrated Probability Score"]:::modelStyle
            LSR_CLF["LSR Classifier<br/>(One-vs-Rest Logistic Regression)<br/>• Output: Primary Rule<br/>• Multi-label Rule Probabilities"]:::modelStyle
        end

        subgraph TrackB["Track B: Entity & Precursor Extraction"]
            EXTRACTOR["SafetyPrecursorExtractor<br/>(src/extraction/precursor_extractor.py)<br/>• Regex entity patterns<br/>• Lexical safety taxonomy"]:::modelStyle
            NORMALIZER["Canonical Normalization Engine<br/>(src/extraction/normalization.py)<br/>• Standardized Activities<br/>• Equipment Catalogs<br/>• Hazards & Barrier Failures<br/>• Consequence Tags"]:::modelStyle
        end
    end

    CLEAN --> EMBED
    CLEAN --> EXTRACTOR
    EMBED --> SIF_CLF
    EMBED --> LSR_CLF
    EXTRACTOR --> NORMALIZER

    %% Enrichment Subsystem
    subgraph S4["4. Intelligence Enrichment & Dataset Assembly"]
        ENRICH["Unified Intelligence Schema<br/>(scripts/build_analytics_dataset.py)<br/>Merges Predictions + Extracted Entities"]:::enrichStyle
        ENRICHED_CSV[("data/analytics/enriched_reports.csv")]:::storageStyle
    end

    SIF_CLF --> ENRICH
    LSR_CLF --> ENRICH
    NORMALIZER --> ENRICH
    ENRICH --> ENRICHED_CSV

    %% Analytics & Mining Subsystem
    subgraph S5["5. Pattern Mining & Risk Aggregation (src/analytics/)"]
        FIM["Frequent Itemset Mining (FP-Growth)<br/>(src/analytics/pattern_mining.py)<br/>Finds recurring multi-factor co-occurrences"]:::analyticsStyle
        PRIORITY["Multi-Criteria Risk Scoring<br/>(src/analytics/risk_scoring.py)<br/>Weights frequency, SIF density, site spread"]:::analyticsStyle
        AGG["Site & Domain Aggregator<br/>(src/analytics/aggregation.py)<br/>• SIF volume & density by site<br/>• Adequate sample thresholding"]:::analyticsStyle
        TEMPORAL["Temporal Trend Engine<br/>(src/analytics/temporal.py)<br/>Quarterly & monthly velocity analysis"]:::analyticsStyle
    end

    ENRICHED_CSV --> FIM
    ENRICHED_CSV --> AGG
    ENRICHED_CSV --> TEMPORAL
    FIM --> PRIORITY

    %% Cache & Persistence Subsystem
    subgraph S6["6. Analytics Cache (artifacts/analytics/)"]
        JSON_SUMMARY[("overall_summary.json")]:::storageStyle
        JSON_SITES[("site_analytics.json")]:::storageStyle
        JSON_LSR[("lsr_analytics.json")]:::storageStyle
        JSON_PATTERNS[("recurring_patterns.json")]:::storageStyle
        JSON_EMERGING[("emerging_patterns.json")]:::storageStyle
        JSON_TEMPORAL[("temporal_analytics.json")]:::storageStyle
        DB_SQLITE[("data/sif_database.sqlite")]:::storageStyle
    end

    PRIORITY --> JSON_PATTERNS
    PRIORITY --> JSON_EMERGING
    AGG --> JSON_SUMMARY
    AGG --> JSON_SITES
    AGG --> JSON_LSR
    TEMPORAL --> JSON_TEMPORAL
    ENRICH --> DB_SQLITE

    %% API Layer
    subgraph S7["7. FastAPI Web Services (src/api/)"]
        API_ROUTER["FastAPI Application (src/api/main.py)<br/>Base URL: /api/v1"]:::apiStyle
        EP_PREDICT["POST /predict & /predict/batch<br/>(Real-time Pipeline Inference)"]:::apiStyle
        EP_ANALYTICS["GET /analytics/{summary, sites, lsr, patterns, ...}<br/>(In-Memory Cached Artifact Serving)"]:::apiStyle
        EP_UPLOAD["POST /analytics/upload<br/>(Dynamic CSV Upload & Background Re-enrichment)"]:::apiStyle
    end

    JSON_SUMMARY -.-> EP_ANALYTICS
    JSON_SITES -.-> EP_ANALYTICS
    JSON_LSR -.-> EP_ANALYTICS
    JSON_PATTERNS -.-> EP_ANALYTICS
    JSON_EMERGING -.-> EP_ANALYTICS
    JSON_TEMPORAL -.-> EP_ANALYTICS

    API_ROUTER --- EP_PREDICT
    API_ROUTER --- EP_ANALYTICS
    API_ROUTER --- EP_UPLOAD

    %% Direct real-time path for ad-hoc prediction
    TrackA -.-> EP_PREDICT
    TrackB -.-> EP_PREDICT

    %% Presentation Layer
    subgraph S8["8. Streamlit Dashboard (src/dashboard/)"]
        UI_OVERVIEW["Executive Overview<br/>(High-level KPIs, Site Volume vs SIF)"]:::uiStyle
        UI_SIF["SIF Analysis<br/>(Probability distribution, High-risk buckets)"]:::uiStyle
        UI_LSR["Life-Saving Rules<br/>(Rule compliance, Cross-site matrix)"]:::uiStyle
        UI_PATTERNS["Precursor Patterns<br/>(FP-Growth itemsets, Priority ranking)"]:::uiStyle
        UI_SITES["Site Risk Ranking<br/>(Adequate sample-weighted densities)"]:::uiStyle
        UI_NEW["Analyse New Report<br/>(Real-time interactive single-text inference)"]:::uiStyle
        UI_UPLOAD["Data Upload<br/>(Batch file ingestion & pipeline trigger)"]:::uiStyle
    end

    EP_ANALYTICS --> UI_OVERVIEW
    EP_ANALYTICS --> UI_SIF
    EP_ANALYTICS --> UI_LSR
    EP_ANALYTICS --> UI_PATTERNS
    EP_ANALYTICS --> UI_SITES
    EP_PREDICT --> UI_NEW
    UI_UPLOAD --> EP_UPLOAD
```

---

## 5. Detailed Component Breakdown

### 5.1 Input Data Ingestion
- **Format Options**:
  - **CSV file** (`data/all_reports.csv` or uploaded via UI) containing structured metadata (`report_id`, `date`, `site`, `location`, `report_type`) along with a free-text `description`.
  - **JSON batch** (`data/demo/demo_reports.json`) containing report objects.
  - **Raw string** via `POST /api/v1/predict` or the Streamlit "Analyse New Report" text area.
- **Example Raw Observation**:
  > *"Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line."*

### 5.2 NLP & Machine Learning Pipeline (`src/pipeline/inference.py`)
When a report passes into `SafetyInferencePipeline`:
1. **Sentence Embeddings**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`.
   - The free-text string is transformed into a dense 384-dimensional semantic representation capturing conceptual safety semantics (e.g., proximity to high energy, missing safety controls).
2. **SIF Classifier** (`src/models/sif_classifier.py`):
   - A scikit-learn binary Logistic Regression classifier trained on embeddings.
   - Evaluates whether the observation constitutes a precursor to a fatal or life-altering event.
   - Outputs: Binary prediction (`0` or `1`), categorical label (`"SIF-Potential"` vs `"Non-SIF"`), and calibrated probability (e.g., `0.8741`).
3. **Life-Saving Rule (LSR) Classifier** (`src/models/lsr_classifier.py`):
   - A multi-class One-vs-Rest (OvR) Logistic Regression model.
   - Evaluates compliance against standardized industry Life-Saving Rules (e.g., *Energy Isolation*, *Work at Height*, *Line of Fire*, *Hot Work*, *Confined Space*).
   - Outputs: `primary_lsr` alongside confidence scores across all candidate rules.
4. **Precursor Entity Extraction** (`src/extraction/precursor_extractor.py` & `normalization.py`):
   - Employs regex patterns, contextual triggers, and safety taxonomies to extract:
     - **Activity**: `Piping / Valve Maintenance`
     - **Equipment**: `Pump Discharge Flange`
     - **Hazard**: `Residual Pressurized Fluid`
     - **Barrier Failure**: `Isolation Not Verified`
     - **Consequence**: `High Pressure Release / Chemical Spray`
   - Canonical normalization maps diverse phrasing to consistent ontology keys.

### 5.3 Enriched Intelligence Dataset (`data/analytics/enriched_reports.csv`)
The pipeline joins the raw metadata with the model inference results to produce an enriched columnar dataset. This format guarantees that downstream analytics consume consistent, reproducible model outputs without repeatedly invoking transformer inference.

### 5.4 Advanced Analytics & Pattern Mining (`src/analytics/`)
1. **Frequent Itemset Mining (FP-Growth)** (`pattern_mining.py`):
   - Converts multi-dimensional precursors (Activity, Hazard, Equipment, Barrier Failure, LSR) into transactional itemsets.
   - Computes support and confidence to uncover combinations that co-occur across reports.
2. **Composite Risk Scoring** (`risk_scoring.py`):
   - Prioritizes patterns using a weighted objective function:
     $$\text{Priority Score} = w_1 \cdot \text{Frequency} + w_2 \cdot \text{SIF Density} + w_3 \cdot \text{Site Spread} + w_4 \cdot \text{Severity}$$
3. **Site Risk Aggregations** (`aggregation.py`):
   - Computes total report volume, SIF counts, and SIF density per operational facility.
   - Implements adequate sample size filters (e.g., minimum report threshold) to prevent small-sample statistical distortions.
4. **Temporal & Emerging Trend Detection** (`temporal.py`):
   - Calculates moving averages and quarter-over-quarter percentage shifts to detect newly emerging risk patterns before serious incidents materialize.

### 5.5 Backend API (`src/api/`)
- Built with **FastAPI** to provide low-latency REST interfaces.
- **In-Memory Cache**: Loads pre-computed analytics JSON artifacts into memory upon startup for sub-millisecond query responses.
- **Dynamic File Ingestion**: The `/api/v1/analytics/upload` endpoint processes uploaded CSV files asynchronously and recalculates analytics.

### 5.6 Streamlit HSE Dashboard (`src/dashboard/`)
- A modular dashboard communicating with the FastAPI backend through `api_client.py`.
- **Pages**:
  - **Executive Overview**: High-level KPIs, total volume vs. SIF rate, and hazard breakdown.
  - **SIF Analysis**: In-depth SIF distributions, confidence score calibration, and type breakdowns.
  - **Life-Saving Rules**: Cross-site LSR matrix and rule-to-hazard associations.
  - **Precursor Patterns**: Interactive table and cards displaying top recurring FP-Growth patterns and component weights.
  - **Site Risk**: Normalized site risk ranking with adequate sample controls.
  - **Activity Analysis**: Deep dive into specific high-risk operational activities.
  - **Emerging Trends**: Fast-moving hazard combinations flagged with percentage increases.
  - **Analyse New Report**: Live interactive sandbox allowing safety engineers to paste raw text and view instant predictions and extracted entities.
  - **Data Upload**: File dropzone to load new observation datasets into the analytics engine.

---

## 6. Execution Quick-Start Reference

To run the complete pipeline locally:

1. **Activate Environment**:
   ```powershell
   venv\Scripts\Activate.ps1
   ```
2. **Generate / Refresh Analytics (Offline Pipeline)**:
   ```powershell
   python scripts/build_analytics_dataset.py
   python scripts/run_analytics.py
   ```
3. **Start FastAPI Backend**:
   ```powershell
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Start Streamlit Dashboard**:
   ```powershell
   streamlit run src/dashboard/app.py
   ```
   *(Or run `.\scripts\start_demo.ps1` to launch both concurrently)*
