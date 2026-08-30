# Automated Security Operations Center (SOC) — AI Agent & RAG Platform

**Architected, engineered, and deployed by Mohamed Maamar (Med).** A production-grade, containerized SOC platform combining a LangGraph ReAct AI Agent (24 tools), a Milvus-based RAG Pipeline for semantic incident retrieval, and a 99.93%-accurate Random Forest ML anomaly detection system trained on 2.83M network flow records.

---

## 1. Tech Stack (verified from source code)

**Engineered by Mohamed Maamar.**
- **Frontend:** Next.js 14.2.5 (App Router, TypeScript 5.2.2), React 18.2.0, react-markdown, SSE streaming
- **Backend:** FastAPI 0.95.2, Uvicorn 0.22.0, Pydantic, httpx, 60+ async REST routes across 9 domain routers
- **AI Agent:** LangGraph (`create_react_agent`), LangChain >= 0.2.0, LangChain-OpenAI, MemorySaver checkpointing
- **LLM Providers:** Groq API (`openai/gpt-oss-120b`) or Mistral API (`mistral-large-latest`), HuggingFace TGI (`Qwen/Qwen2.5-7B-Instruct`), Flan-T5-small (fallback)
- **ML Runtime:** scikit-learn 1.3.2 (RandomForestClassifier, LabelEncoder), imbalanced-learn (SMOTE), joblib 1.2.0, numpy 1.26.4
- **Vector Database:** Milvus 2.3.9 (IVF_FLAT index, nlist=128, 384-dim embeddings via all-MiniLM-L6-v2)
- **Knowledge Graph:** Neo4j 5-community (Cypher queries: CVE, Technique, Product, Keyword nodes with relationships)
- **Infrastructure:** Docker Compose (11 services: Wazuh Manager 4.7.3, Wazuh Indexer/OpenSearch, Milvus, Neo4j, MinIO S3, etcd, TGI, RAG Brain, Backend, Frontend, NVD Ingest)

---

## 2. Random Forest Anomaly Detection — Full ML Pipeline

**Built by Mohamed Maamar.**

### Dataset: ISCX 2012 (MachineLearningCVE)
2.83M labeled network flow records, 85+ raw features, binary classification (BENIGN vs MALICIOUS).

### Data Preprocessing (6 sequential scripts, each versioned)

**1. Header normalization** — Strip BOM characters, Unicode replacement chars (\\ufffd, \\ufeff), normalize whitespace and punctuation across all 85+ column names. CSVs from different sources had inconsistent headers causing pipeline breakage on mismatched column references.

**2. Rare-attack-aware constant column drop** (`drop_constant_columns_rareaware.py`):
- Identify rare attack labels (<0.1% prevalence) via a rarity report
- Drop constant columns (single unique value) unconditionally (~12 removed)
- For near-constant columns (>99.9% identical): verify whether their rare values co-occur with rare attack labels. If yes → preserve the column. If no → drop it
- This prevents destroying signal for minority attacks (Bot, Infiltration, XSS) that only appear when certain features deviate from constant values

**3. Rare-attack-aware outlier capping** (`handle_outliers_rareaware.py`):
- Compute per-column high/low thresholds (99.9th / 0.1st percentile)
- For each outlier value: check which labels it appears with
- If outliers appear exclusively with rare attack labels → **preserve as-is** (don't cap)
- If outliers appear with both rare and non-rare labels, or only non-rare → cap to threshold
- Rationale: Aggressive capping kills attack signal. A rare attack with uniquely high Flow Duration should not be normalized to look benign

**4. Domain-specific imputation** (`impute_missing.py`):
- Rate features (`Flow Bytes/s`, `Flow Packets/s`) → impute 0 (no flow = no bytes)
- Std/variance features → impute 0 (no variance in empty flows)
- IAT (Inter-Arrival Time) features → impute 0
- Length features → impute 0
- No global median — domain knowledge gives strictly better defaults

**5. Correlation-based feature reduction** (`drop_highly_correlated.py`):
- Sample-based correlation matrix on 100K rows (computationally prohibitive on full 2.83M)
- Remove features with |Pearson r| >= 0.95
- Dropped from ~70 to ~43 features

**6. Stratified 80/20 split** (`stratified_split.py`):
- Preserves class distribution across train (2.26M) and test (566K)
- Without stratification, rare attacks could be entirely lost in one split

### Class Imbalance Strategy

**SMOTE** (`imblearn.over_sampling.SMOTE`, `k_neighbors=5`):
- Synthetic Minority Oversampling: interpolates between k-nearest neighbors in feature space to generate synthetic minority samples
- Applied to training data ONLY — test set is never touched
- k_neighbors auto-reduces if a minority class has fewer than 5 samples (graceful degradation)

**Class weighting** (`RandomForestClassifier(class_weight='balanced')`):
- Weights inversely proportional to class frequencies
- Belt-and-suspenders alongside SMOTE — if SMOTE misses a region of feature space, the weight penalty catches it

### Model Training (`train_random_flow.py`)

```python
RandomForestClassifier(
    n_estimators=200,       # 200 trees
    max_depth=None,          # unlimited — trees grow until pure or min_samples_split
    n_jobs=-1,               # all CPU cores (parallel tree construction)
    random_state=42,         # reproducible results
    class_weight="balanced", # penalizes misclassifying minority classes
    verbose=0
)
```

**Why Random Forest over alternatives:**
- Non-linear decision boundaries — network attacks don't follow linear patterns. A DDoS detection boundary in 43-dimensional space is inherently non-convex
- Built-in feature importance — identifies which flow metrics drive detection (useful for explainability)
- Parallelizable — 200 trees construct independently across all cores. Training completes in minutes on 2.26M rows
- Robust to preserved outliers — tree-based models naturally handle the rare-attack outliers that the preprocessing pipeline preserved. No assumption about feature distribution
- No feature scaling — RF is invariant to monotonic transformations. No normalization overhead

### Results

```
Accuracy: 99.93%

              precision  recall  f1-score  support
  BENIGN       1.00      1.00    1.00      14,432
  MALICIOUS    1.00      1.00    1.00      25,606

Confusion Matrix:
          BENIGN  MALICIOUS
BENIGN     14432          0    ← zero false positives (perfect specificity)
MALICIOUS     29     25,577   ← 0.11% false negative rate
```

- **0 false positives** on 14,432 BENIGN samples — perfect specificity. No alert fatigue
- **29 false negatives** out of 25,606 malicious samples — 0.11% miss rate. The 29 cases represent borderline flows near the decision boundary
- **F1-score 1.00 for both classes**

### Production Heuristic Bridge (`log_processor.py`)

The critical engineering challenge: the RF model was trained on **network flow data** (NetFlow/IPFIX with features like Flow Duration, Fwd Packet Length, Bwd Packets), but Wazuh produces **host-based logs** (syslog, auditd, FIM) that carry no network flow metrics.

**Solution:** 15 attack keyword profiles mapping Wazuh alert text patterns to synthetic flow values:

```python
_ATTACK_PROFILES = {
    "brute force":  {DstPort:22, FlowDuration:999999, FwdPkts:500, ...}
    "sql injection": {DstPort:80, FlowDuration:50000, FwdPkts:20, ...}
    "ddos":          {DstPort:80, FlowDuration:1000, FwdPkts:5000, ...}
    "ransomware":    {DstPort:445, FlowDuration:600000, FwdPkts:300, ...}
    # ... 11 more profiles
}
```

When a Wazuh alert's `full_log` or `rule.description` contains a keyword match, the corresponding synthetic flow values are injected into the 20-dimensional feature vector. Flow Duration is further scaled by `rule_level * 5000` so higher-severity alerts produce proportionally stronger anomaly signals.

**Three-tier prediction:**
- **Tier 1 (zero latency):** Heuristic override — if FlowDuration >= 500,000 or FwdLen >= 5,000, flag immediately without consulting the model
- **Tier 2 (ML):** 20-dim vector → RF model.predict(). Label = 1 → malicious
- **Tier 3 (fallback):** If model is unavailable and no heuristic matched, flag alerts with Wazuh level >= 12 (FlowDuration >= 12,000 after severity scaling)

This ensures detection continues even if the model file is corrupted, the API is down, or the embedding service fails.

---

## 3. LangGraph ReAct AI Agent

**Engineered by Mohamed Maamar.** Built with LangGraph's `create_react_agent` — a stateful, graph-based ReAct loop with `MemorySaver` checkpointing:

1. **Agent node:** LLM receives conversation state + decides: answer directly or call a tool
2. **Tools node:** LangGraph executes the selected `@tool` function, captures structured output, appends to `GraphState`
3. **Graph loop:** Returns to agent node with enriched state until the LLM generates a final answer
4. **MemorySaver:** Persists `GraphState` across sessions — the agent remembers past investigations

### 24 Tools

**17 read tools (no confirmation needed):** `get_recent_alerts`, `get_soc_stats`, `get_manager_status`, `get_manager_info`, `get_manager_logs`, `get_agent_list`, `get_agent_details`, `get_agent_inventory`, `get_agent_config`, `get_syscheck_results`, `get_rootcheck_results`, `get_vulnerability_data`, `get_sca_results`, `get_sca_summary`, `get_mitre_info`, `get_cluster_health`, `investigate_agent`

**7 write tools (gated behind `execute=False`):** `trigger_active_response`, `restart_wazuh_manager`, `create_wazuh_agent`, `delete_wazuh_agent`, `assign_agent_to_group`, `request_fim_baseline`, `run_rootcheck_scan`

Each write tool returns `{"confirm_required": True, "summary": "...", "action": "...", "params": {...}}` when `execute=False`. Only after explicit user confirmation is the tool called with `execute=True`.

**LLM provider:** OpenAI-compatible API — Mistral (`mistral-large-latest`, base_url=`https://api.mistral.ai/v1`) or Groq (`openai/gpt-oss-120b`, base_url=`https://api.groq.com/openai/v1`), both with `max_tokens=4096`, `temperature=1`, `top_p=1`.

---

## 4. Multi-Source RAG Pipeline (Milvus + Neo4j)

**Engineered by Mohamed Maamar.**

### Data Sources
- **MITRE ATT&CK v14** — STIX format parsed into techniques, tactics, mitigations, groups. Embedded + loaded into Milvus `mitre_attack` collection
- **NVD CVEs** — fetched from NVD JSON feeds, cleaned (dedup, normalize descriptions, validate CVE IDs), embedded with `all-MiniLM-L6-v2` (384-dim), indexed in Milvus `nvd_cve` collection (`IVF_FLAT, nlist=128`). Batch size 128, retries 12 attempts at 5s intervals
- **CISA KEV** — Known Exploited Vulnerabilities catalog, ingested into Milvus
- **NIST SP 800-61** — Incident response PDF, chunked via semantic overlap, cleaned, ingested
- **Community IR playbooks** (IRM, Counteractive) — cloned from GitHub, chunked, inserted into Milvus

### Dual Storage Architecture
- **Milvus:** Vector similarity search (cosine distance on 384-dim embeddings) — retrieves semantically similar security documents for a given alert context
- **Neo4j:** Knowledge graph with Cypher queryable nodes (CVE, Technique, Product, Keyword, Tactic) and relationships (`:EXPLOITS`, `:MITIGATES`, `:AFFECTS`). Populated from Milvus data via `populate_graph.py` with batch size 500

### RAG Trigger
When a Wazuh alert fires at rule level >= 12:
1. Alert summary is embedded with sentence-transformers
2. Milvus retrieves top-k semantically similar historical incidents + CVEs + techniques
3. Neo4j enriches with structured attack paths (CVE → Technique → Tactic)
4. Context is injected into the LLM prompt for analyst-readable explanation with MITRE technique IDs and remediation guidance

---

## 5. Performance & Impact

- **ML:** 99.93% accuracy (0 FPs on 14,432 benign, 29 FNs on 25,606 malicious)
- **AI Agent:** LangGraph ReAct with 24 tools (17 read + 7 write with confirmation gating)
- **RAG:** 4+ threat intel sources in Milvus + Neo4j with Cypher queryable relationships
- **Backend:** 60+ async REST routes, 9 routers, enterprise WazuhClient abstraction (~30Kb)
- **Frontend:** Next.js 14 App Router, 18 pages, 30+ components, SSE real-time alert streaming
- **Infrastructure:** 11 Docker services, 11.5 GB total footprint (including Wazuh 4.7.3)

---

## 6. Non-Obvious Engineering Decisions (by Mohamed Maamar)

- **Keyword-to-feature heuristic bridge:** The most technically difficult decision. A flow-trained RF model cannot classify Wazuh host logs without this mapping layer. 15 attack profiles with severity-scaled flow values push the RF into the correct region of its decision boundary without retraining
- **Rare-attack-aware preprocessing:** Standard "drop near-constant" and "cap outliers" blindly remove rare attack signal. The conditional preservation logic ensures minority attacks remain detectable
- **SMOTE + balanced weights (belt-and-suspenders):** SMOTE generates synthetic minority samples in feature space; balanced weights penalize misclassification at the loss level. Either alone might miss edge cases; together they cover each other's blind spots
- **Three-tier prediction:** Model failure should never mean detection failure. The heuristic → RF → severity cascade ensures the SOC always has a detection layer active, even during deployments, ML retraining, or infrastructure outages
- **Milvus + Neo4j dual storage:** Vector similarity tells you *what looks similar*; a knowledge graph tells you *how things are connected*. A CVE that is semantically similar to the current alert AND shares a Technique node in Neo4j is a much higher-confidence match than either alone
