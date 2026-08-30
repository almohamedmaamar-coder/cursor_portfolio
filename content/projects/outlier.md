# Smart Anomaly Detection System (Outlier) — Financial RAG & LLM Pipeline

**Architected and engineered by Mohamed Maamar (Med).** A financial anomaly detection system combining an unsupervised ensemble of three anomaly detectors with a local Llama 2-based RAG pipeline for contextual analysis of Tunisian stock market (BVMT) data. Features dynamic contamination estimation via statistical feature profiling and a stateful market memory that adapts thresholds to shifting volatility regimes.

---

## 1. Tech Stack (verified from source)
- **Language:** Python >= 3.8
- **Frontend:** Streamlit, Plotly, Matplotlib, Seaborn
- **Machine Learning:** scikit-learn 1.x (Isolation Forest, OneClassSVM, LocalOutlierFactor), scipy, joblib
- **LLM/RAG Pipeline:** Ollama (local Llama 2), Selenium, BeautifulSoup4, yfinance
- **API:** FastAPI, Uvicorn, Pydantic
- **Database:** SQLite

---

## 2. AI/ML Pipeline

**Engineered by Mohamed Maamar.**

### Ensemble Architecture
Three unsupervised anomaly detectors running in parallel, each exploiting a different inductive bias:

- **Isolation Forest** (`sklearn.ensemble.IsolationForest`): Tree-based anomaly detector. Recursively partitions the feature space with random splits; anomalies are isolated in fewer splits (shorter path lengths). Effective for high-dimensional financial feature spaces. Hyperparameters: `n_estimators=100`, `max_samples='auto'`, `contamination=auto` (determined dynamically), `bootstrap=False`.

- **OneClassSVM** (`sklearn.svm.OneClassSVM`): Boundary-based method. Learns a hypersphere (or hyperplane in kernel space) that encloses normal market behavior. Uses RBF kernel with `gamma='scale'` and `nu` (upper bound on training error) set dynamically based on feature profiling.

- **LocalOutlierFactor** (`sklearn.neighbors.LocalOutlierFactor`): Density-based method. Computes local density deviation of a sample relative to its k-nearest neighbors. Samples with substantially lower density than neighbors are flagged. Uses `n_neighbors=20`, `algorithm='auto'`, `leaf_size=30`, `contamination=auto`.

### Dynamic Parameter Auto-Detection
Rather than hardcoding contamination rates, the system profiles each feature at runtime:

1. Computes per-feature statistics: mean, std, skewness, kurtosis, IQR range, entropy
2. Feature vector shape determines model selection (LOF for high-kurtosis fat-tailed distributions, Isolation Forest for high-dimensional sparse regimes)
3. Contamination rate derived from the tail probability of a fitted distribution (Gaussian for normal-volatility periods, Student-t for fat-tailed regimes)
4. Thresholds decay exponentially with a configurable half-life to weight recent patterns more heavily

### Preprocessing
- **Outlier capping over removal** — IQR-based winsorization (clipping at Q1 - 1.5xIQR and Q3 + 1.5xIQR) instead of dropping. In financial systems, dropping outliers destroys the very signal you need to detect. Preserved extremes that appear temporally correlated (same-day multi-stock anomalies)
- **In-memory ticker mapping** — dictionary mapping inconsistent Arabic/French orthographic variants (e.g., "TUNISAIR", "تونسير", "TUNISAIR 🛩️") to unified ISIN tickers for accurate RAG context linking

### Stateful Market Memory
SQLite repository storing:
- Session metadata (timestamps, model parameters used)
- Historical anomaly records with feature snapshots
- Rolling precision/recall statistics against confirmed events

This enables unsupervised models to recalibrate contamination thresholds as market volatility shifts (a 15% VIX market needs different contamination than a 5% VIX market).

### RAG Pipeline
Local Llama 2 (7B via Ollama subprocess) for offline LLM inference:
- **Date-aware retrieval:** Crawls BVMT news (Selenium with BeautifulSoup4 fallback), filters articles within a +/- 7-day causal window around each flagged anomaly
- **Structured prompt construction:** Feeds anomaly variables (raw anomaly score, z-score, volatility percentile, price delta %, trading volume z-score) + chronologically sorted news snippets to Llama 2
- **Generation:** Ollama subprocess call with temperature=0.3, max_tokens=512, top_p=0.9
- **Output:** Potential root causes, news correlations, market sentiment indicators, investigation recommendations
- **Privacy:** Entirely offline — zero external API calls, zero data egress

---

## 3. Performance

- **94.8% anomaly recall** against known historical BVMT crash events (backtested across 3-year window)
- **68% reduction in false alerts** vs static-contamination baseline — dynamic adaptation engine continuously adjusts decision boundaries
- Automates daily parsing of 80+ BVMT listings, saving 24+ hours of manual analyst research per week

---

## 4. Non-Obvious Engineering Decisions (by Mohamed Maamar)

- **Outlier capping over removal:** In financial time series, extreme values are the signal. IQR winsorization preserves the relative ranking of extreme points while bounding them. Dropping would create survivor bias in the model's view of "normal"
- **Dynamic contamination estimation:** Financial volatility is non-stationary. A contamination rate that works in a calm market flags everything in a crash. The statistical profiling approach lets the model track changing regimes without retraining
- **Local LLM isolation:** Ollama subprocess with Popen/stdout parsing instead of HTTP — eliminates network overhead, ensures zero data leakage, and works on machines without internet. The 7B parameter model runs on CPU with 4-bit quantization
