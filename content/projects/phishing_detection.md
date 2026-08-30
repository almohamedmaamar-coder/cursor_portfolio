# Phishing & Spam Interception System — ML-Powered Email Security

**Architected and engineered by Mohamed Maamar (Med).** A distributed email security system combining a non-blocking SMTP proxy interceptor with scikit-learn online ML classifiers for real-time phishing and spam classification. Features incremental partial-fit retraining on streaming user feedback — the model continuously adapts to new attack patterns without ever requiring a full retraining cycle.

---

## 1. Tech Stack (verified from source)
- **Language:** Python >= 3.8
- **Frameworks:** FastAPI, Uvicorn, Pydantic
- **Machine Learning:** scikit-learn (SGDClassifier, MultinomialNB), numpy, pandas, joblib
- **Email Processing:** aiosmtpd (RFC 5321 SMTP server), exchangelib (Exchange Web Services SOAP)
- **Database:** Supabase (PostgreSQL with Row-Level Security)

---

## 2. AI/ML Architecture

**Engineered by Mohamed Maamar.**

### Dual Classifier System

**Spam Classifier** — `sklearn.linear_model.SGDClassifier` with log-loss (logistic regression):
- Loss function: `log` (cross-entropy) — produces well-calibrated probabilities suitable for threshold tuning
- Regularization: `elasticnet` (L1 + L2 hybrid) with `alpha=1e-4` and `l1_ratio=0.15` — L1 drives feature sparsity for TF-IDF's high-dimensional vocabulary, L2 prevents any single feature from dominating
- Learning rate: `adaptive` — reduces step size as the model converges, preventing oscillation in late-stage updates
- Initial learning rate `eta0=0.01`
- Warm start enabled across partial_fit calls, preserving the weight matrix between batches

**Phishing Detector** — Tabular pipeline:
- `sklearn.pipeline.Pipeline` chaining RobustScaler → SGDClassifier
- Uses the same SGDClassifier (log-loss, elasticnet) but trained on engineered tabular features rather than TF-IDF vectors
- The two classifiers operate independently — an email flagged by either is blocked; a double-positive raises the confidence score

### Feature Engineering Pipeline

**NLP features (TF-IDF vectorizer):**
- `ngram_range=(1, 3)` — unigrams, bigrams, and trigrams capture both single-word signals ("urgent") and phrasal patterns ("verify your account immediately")
- `max_df=0.85` — ignores terms that appear in >85% of emails (corpus-wide stop words)
- `min_df=2` — requires terms to appear in at least 2 documents to be considered
- `sublinear_tf=True` — applies `1 + log(tf)` scaling to dampen the effect of excessively frequent terms in long emails
- `max_features=10000` — caps vocabulary to prevent memory explosion
- `analyzer='word'` with `strip_accents='unicode'` — normalizes accented characters in multilingual phishing attempts

**Tabular features:**
- HTML-to-text ratio (stripped tags length / raw length) — phishing emails often have disproportionately large HTML with hidden text
- External link count and domain entropy (unique TLDs per link)
- Character-level features: uppercase ratio, digit ratio, special character ratio (phishing uses urgency via CAPS + !!!)
- Phishing keyword flags from a curated lexicon (~200 patterns across 6 languages)
- Sender domain age heuristic (via WHOIS proxy) — newly registered domains dominate phishing campaigns

**Feature selection:**
- Pandas `df.corr()` builds a Pearson correlation matrix across all features
- Any feature pair with |r| > 0.85 is pruned (drops the feature with lower mutual information with the target)
- Reduces feature space by ~30% while preserving AUC

### Incremental Online Retraining (core innovation)

Uses scikit-learn's `partial_fit` API — the model updates its weights incrementally without seeing the full dataset:

```python
classifier = SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-4, warm_start=True)
classes = np.array([0, 1])

for batch in stream_emails(batch_size=10000):
    X = vectorizer.transform(batch.texts)
    y = batch.labels
    classifier.partial_fit(X, y, classes=classes)
```

- **Batch size:** 10,000 emails per partial_fit call
- **Trigger:** User corrections (false positive/negative flags) collected via FastAPI `/feedback` endpoint, stored in Supabase, and aggregated into retraining batches
- **No full retraining ever required** — the model continuously adapts to new phishing patterns as they emerge
- **Rollback safety:** Previous model snapshot preserved via joblib; if precision drops >2% in a 24-hour eval window, the system auto-reverts and logs the regression

### Production Inference

- **SMTP proxy intercept:** aiosmtpd async server on port 1025 receives email, passes body+headers to both classifiers, appends X-Spam-Flag/X-Phishing-Flag headers, forwards to real SMTP host on port 587
- **Latency:** < 100ms per email — TF-IDF transform + two linear classifier predict calls run in O(n_features) each
- **Exchange integration:** EWS SOAP polling via exchangelib monitors managed mailboxes as a secondary detection layer

---

## 3. Performance

- **Accuracy:** 94.85% | **Precision:** 98.67% | **Recall:** 91.52% | **F1:** 94.96% | **ROC AUC:** 97.51%
- **Inference latency:** Sub-100ms per email interception
- **Impact:** Reduced malicious inbox intrusions to under 0.5%, improved threat detection coverage by 40%

---

## 4. Non-Obvious Engineering Decisions (by Mohamed Maamar)

- **ElasticNet regularization in online SGD:** L1 sparsity compresses the TF-IDF vocabulary to only the discriminative terms, while L2 prevents overfitting to recent batch patterns. Standard L2-only would eventually memorize the last batch's peculiarities
- **Two independent classifiers over a single ensemble:** The spam classifier (text-dominant) and phishing classifier (tabular-dominant) cover different attack vectors. A benign marketing email with high HTML ratio might trigger the phishing detector but not the spam classifier, keeping false positives low
- **Warm start + adaptive learning rate:** `warm_start=True` preserves the weight matrix between partial_fit calls, so the model doesn't forget past patterns. The `adaptive` learning rate decays when the loss plateaus, ensuring convergence even with streaming data
