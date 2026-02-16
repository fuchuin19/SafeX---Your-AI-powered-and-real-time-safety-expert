### GPT LINK: https://chatgpt.com/share/e/699343f0-ba54-8008-8d38-6efcebf43385 

# SafeX Option A Android Integration Guide (TF‑IDF + Logistic Regression)

> Audience: teammate implementing on‑device scam detection.
>
> Goal: Integrate the **existing** SafeX Track A model into Android **without TFLite** by shipping TF‑IDF assets + Logistic Regression weights and reproducing the same preprocessing and inference steps on device.
>
> **No hallucinations note:** Everything below is derived from the repo’s current notebook/scripts + deployment docs/spec files. Where the repo is ambiguous or inconsistent, it is explicitly called out.

---

## 1) What Option A is (current model you tested)

### 1.1 Model type

* **TF‑IDF vectorizer** (word n‑grams) + **Logistic Regression** classifier.
* Inference uses `predict_proba` (probability of scam) then applies **MID/HIGH thresholds** to choose a band.

### 1.2 Track A vs Track B confusion (important)

* Repo docs mention “Local Model (TFLite)” in architecture diagrams, but the actual Option A path is **manual TF‑IDF + LR math**, not a `.tflite` model.
* There is also a **Track B** mobile‑oriented spec (smaller features, extra scalars, optional quantization). Option A is what you’re running now (sklearn TF‑IDF + LR).

**Action:** For this guide, we implement **Option A** only.

---

## 2) Deliverables you must produce for Android (assets)

### 2.1 Required model assets (Option A)

Place under `app/src/main/assets/`:

1. `tfidf_vocab.json`

   * Maps **feature string** → **index** (size ≈ `max_features`).
2. `tfidf_idf.json`

   * Array/dict of IDF values aligned to vocab indices.
3. `model_coefficients.npy` *(or `model_coef.npy` depending on export)*

   * Logistic Regression weights vector aligned to vocab indices.
4. `model_intercept.npy`

   * Logistic Regression intercept (bias term).

> Repo mismatch to resolve:
>
> * Notebook export uses `model_coef.npy`.
> * Some docs expect `model_coefficients.npy`.
>
> **Choose one name and keep it consistent** across export + Android loader.

### 2.2 Required routing/spec assets

Also place under `app/src/main/assets/`:

5. `thresholds.json`

   * Contains `high_threshold`, `mid_threshold`, and routing policy.
6. `rule_overrides.json`

   * Regex/keyword rule set that can force HIGH risk for extreme patterns.
7. `preprocess_spec.json`

   * The source of truth for placeholder patterns + normalization steps.
8. `label_map.json`

   * For mapping class indices to human labels.

### 2.3 Optional explainability assets (nice for demo)

9. `top_features.json`

   * Precomputed top features for scam/benign.
10. `rationale_templates.json`

* Template explanations to show users “why flagged”.

---

## 3) Generate the model assets (Python export)

### 3.1 Run the training notebook/script

* Run the repo’s Kaggle notebook script (e.g., `SafeX_Kaggle_Notebook.py`) end‑to‑end until the **export section**.

### 3.2 Confirm Track A TF‑IDF settings

Before exporting, confirm the TF‑IDF vectorizer is configured as Track A:

* `ngram_range = (1, 2)`
* `max_features = 40000`

> Repo inconsistency:
>
> * `preprocess_spec.json` shows TF‑IDF params with `max_features = 20000` (mobile‑leaning).
> * The Track A notebook config uses `max_features = 40000`.
>
> **For Option A Android integration**, match the model you trained/exported. If you export a 40k model, Android must load/compute a 40k vector.

### 3.3 Export files

The export step should produce:

* `tfidf_vocab.json`
* `tfidf_idf.json`
* `model_coef.npy`
* `model_intercept.npy`

Then either:

* Rename `model_coef.npy` → `model_coefficients.npy`, **or**
* Update Android code to read `model_coef.npy`.

### 3.4 Thresholds schema alignment

There are two possible schemas:

* Notebook exports a minimal schema like `{ "mid": ..., "high": ... }`.
* Repo’s shipped `thresholds.json` includes `mid_threshold`, `high_threshold`, and a `routing_policy` section.

**Recommended:** Ship the repo’s `thresholds.json` format (more expressive). Adjust training export accordingly if needed.

---

## 4) Android implementation overview (pipeline)

**Order of operations (must match training/inference intent):**

1. **Rule overrides (fast, high‑risk patterns)**
2. **Text preprocessing** (normalize + placeholder masking + CJK spacing)
3. **TF‑IDF vectorization** (word 1–2 grams)
4. **Logistic Regression scoring** (dot + intercept + sigmoid)
5. **3‑band routing** using thresholds
6. (Optional) server fallback for MID band

---

## 5) Implement text preprocessing (most important for parity)

> Most integration failures come from preprocessing mismatches.

### 5.1 Required preprocessing steps

Implement a Kotlin `cleanText(raw: String): String` that mirrors the notebook’s `clean_text()`.

**Must include:**

1. **Strip code fences / generator artifacts**

* Remove blocks like triple backticks and language tags.

2. **Remove “meta lines”**

* Lines beginning with patterns like:

  * `Tetapan:`
  * `Nada:`
  * `Konteks:`
  * `Teknik:`
  * `Platform:`

3. **Normalize obfuscation**

* Remove zero‑width chars
* Collapse spaced letters / dotted words e.g. `W.h.a.t.s.A.p.p`

4. **Normalize placeholders (privacy + model parity)**
   Replace in this order (recommended):

* URLs → `<URL>`

  * `https?://\S+`
  * `www\.\S+`

* Phone → `<PHONE>`

  * Malaysia‑style patterns (at minimum handle `01x-xxxxxxx` and general digit sequences)

* Email → `<EMAIL>`

* RM amounts → `<RM_AMOUNT>`

  * `RM\s?\d+(?:[\.,]\d+)?` and variants

* Dates → `<DATE>`

  * Simple date formats used in synthetic templates

* (If used in training) OTP/TAC → `<OTP>`

  * Only if the training cleaner does this. If not, do not introduce it.

5. **Add CJK spacing**

* Insert spaces between Chinese characters (or use char‑ngram fallback).
* This helps TF‑IDF tokenization treat Han characters sensibly.

6. **Whitespace normalization**

* Lowercase
* Collapse multiple spaces
* Trim

### 5.2 Keep preprocessing spec as the single source of truth

* Load `preprocess_spec.json` and implement the listed steps.
* Where `preprocess_spec.json` is missing a regex that exists in the notebook, prioritize the notebook (because that’s what produced the trained model).

**Action:** Write a unit test that compares Kotlin `cleanText()` output with Python `clean_text()` for a list of messages.

---

## 6) TF‑IDF vectorization (must match training)

### 6.1 What you must match

* **Word n‑grams: (1,2)** (unigrams + bigrams)
* **Feature space size:** equals vocab size (e.g., 40,000)
* **Sublinear TF:** if training used it, apply `tf = 1 + ln(count)`

### 6.2 Implementation approach (sparse)

Do **not** allocate a dense 40k float array per message if you can avoid it.

Recommended approach:

* Build a `MutableMap<Int, Double>` (or Int2Double structure) of featureIndex → tfidfValue.
* Compute only for features present:

  1. Tokenize cleaned text into tokens list.
  2. Emit unigrams and bigrams:

     * unigram: `tokens[i]`
     * bigram: `tokens[i] + " " + tokens[i+1]`
  3. Count occurrences in a `Map<String, Int>`.
  4. For each n‑gram string:

     * if present in `vocab`, get index
     * compute `tf = 1 + ln(count)` (if sublinear)
     * multiply by `idf[index]`
     * store in sparse map.

### 6.3 Tokenization caveat (critical)

Scikit‑learn’s tokenization may differ from naive whitespace splitting.

To avoid drift:

* Try to match sklearn defaults as closely as possible (lowercasing, stripping punctuation).
* If performance allows, implement a regex tokenization similar to sklearn’s default `token_pattern` (word characters of length >=2).

**Action:** Validate with golden tests (Section 9).

---

## 7) Logistic Regression scoring

### 7.1 Math

Given sparse TF‑IDF features `x_i`, weights `w_i`, intercept `b`:

* `logit = b + Σ (x_i * w_i)`
* `prob = sigmoid(logit) = 1 / (1 + exp(-logit))`

### 7.2 Kotlin implementation

* Load `model_coefficients.npy` into a FloatArray/DoubleArray `w`.
* Load `model_intercept.npy` into `b`.
* Compute dot product only over non‑zero features (sparse map):

  * `sum += tfidfValue * w[index]`

---

## 8) Risk routing policy (3 bands)

### 8.1 Load thresholds from `thresholds.json`

* `high_threshold`
* `mid_threshold`

### 8.2 Apply banding

* If rule override triggers → `HIGH` (force warn)
* Else:

  * `prob >= high_threshold` → `HIGH`
  * `prob >= mid_threshold` → `MID`
  * else → `LOW`

### 8.3 UI/behavior (offline‑first)

* **HIGH:** strong warning, block / interrupt flow, show rationale.
* **MID:** soft warning; optionally call server fallback (Gemini/Functions) for verification.
* **LOW:** no warning.

---

## 9) Rule overrides (high‑risk patterns)

### 9.1 Load `rule_overrides.json`

Rules cover categories such as:

* Govt aid + URL + sensitive terms (bank/otp/tac/password)
* E‑commerce refund + URL + transfer request
* Customs fee + money request
* Chinese mule‑account keywords (often with money)
* Investment “syariah/halal” + guaranteed return + urgency

### 9.2 Implementation

Create `RuleOverrideEngine`:

* Input: raw text + cleaned text + simple derived flags (has_url, has_money, etc.)
* For each rule:

  * check keyword/regex conditions
  * if match → return `HIGH` with rule id + rationale

Run before model to ensure **high recall** for extreme scams.

---

## 10) Asset loading in Android

### 10.1 When to load

* Load once at app startup (or first use) and cache singleton.
* Use lazy loading and background thread to avoid UI jank.

### 10.2 What to load

* `tfidf_vocab.json` → Map<String, Int>
* `tfidf_idf.json` → DoubleArray aligned to vocab indices
* `model_coefficients.npy` → DoubleArray aligned to vocab indices
* `model_intercept.npy` → Double
* `thresholds.json` → mid/high
* `rule_overrides.json` → list of rules
* `preprocess_spec.json` → patterns + toggles

### 10.3 NPY parsing note

`.npy` parsing is not standard in Kotlin.

**Two safe options:**

1. Implement a small `.npy` reader (NumPy v1.0 header + little‑endian float32/float64).
2. Convert `.npy` to JSON/CSV during build/export and load JSON.

Pick one and document it.

---

## 11) Testing & parity verification (must do)

### 11.1 Golden tests (required)

Create a Python script that outputs a golden file:

* Input raw messages
* Output:

  * `clean_text` result
  * `prob_scam`
  * `band`

Then in Android unit tests:

1. Assert Kotlin `cleanText(raw)` equals Python `clean_text(raw)`.
2. Compute `prob` and compare with small tolerance.
3. Assert band matches.

### 11.2 Stress tests

* Test Malay/Manglish conversational benign messages (to avoid false positives).
* Test Chinese benign + Chinese scam.
* Test URLs (masked), phone numbers, RM amounts.

---

## 12) Performance guidance (keep latency low)

### 12.1 Expected latency

Option A can be fast, but only if you implement sparsely and avoid allocations.

### 12.2 Best practices

* Use sparse maps for TF‑IDF.
* Avoid per‑token object creation.
* Reuse buffers.
* Cache vocab map and IDF arrays.
* Keep preprocessing regex compiled.

### 12.3 Measure on device

Implement a simple benchmark:

* Run inference on 200–500 messages
* Measure avg / p95 latency
* Monitor memory

---

## 13) Integration checklist (hand this to teammate)

### A) Training/export

* [ ] Run training script/notebook to produce TF‑IDF + LR model.
* [ ] Confirm TF‑IDF params match the intended model (40k, 1–2 grams).
* [ ] Export: `tfidf_vocab.json`, `tfidf_idf.json`, `model_coef.npy`, `model_intercept.npy`.
* [ ] Rename/align coefficient filename.
* [ ] Ensure `thresholds.json` schema matches what app expects.

### B) Android assets

* [ ] Copy model assets into `app/src/main/assets/`.
* [ ] Copy `thresholds.json`, `rule_overrides.json`, `preprocess_spec.json`, `label_map.json`.
* [ ] (Optional) Copy `top_features.json`, `rationale_templates.json`.

### C) Android code

* [ ] Implement `cleanText()` matching Python.
* [ ] Implement TF‑IDF with unigrams + bigrams.
* [ ] Implement LR scoring (dot + sigmoid).
* [ ] Implement rule overrides (force HIGH).
* [ ] Implement routing (LOW/MID/HIGH).
* [ ] Add MID server fallback hook (optional).

### D) Verification

* [ ] Build golden test set from Python.
* [ ] Add Android parity unit tests.
* [ ] Benchmark latency.

---

## 14) Known repo gaps / things you MUST decide

1. **Tokenization parity**

* sklearn tokenization differs from naive splitting; you must choose a Kotlin tokenizer and validate.

2. **`.npy` loading strategy**

* Decide: implement NPY reader vs convert to JSON.

3. **Threshold file schema**

* Decide: use repo’s `thresholds.json` schema or notebook’s minimal schema; keep one.

4. **Track A vs Track B**

* This guide is for Option A. If you later move to Track B (structured features + smaller vocab), you will need `scaler.json` and different feature assembly.

---

## 15) Minimal Kotlin class/module structure (suggested)

* `ModelAssetsLoader`

  * loads vocab/idf/weights/intercept/thresholds/rules/spec

* `TextNormalizer`

  * implements `cleanText(raw)`

* `TfidfVectorizerLite`

  * emits sparse features from `cleanText` output

* `LogRegScorer`

  * computes `prob`

* `RuleOverrideEngine`

  * checks `rule_overrides.json`

* `RiskRouter`

  * applies thresholds + outputs `{band, prob, triggeredRule?}`

* `ScamDetector`

  * orchestrates full pipeline

---

## 16) What to send me if you want a final correctness check

If your teammate wants me to sanity-check parity and performance, paste:

* Kotlin `cleanText()` implementation
* Kotlin tokenizer + n‑gram generator
* How you compute sublinear tf
* How you load NPY/weights

Then I can point out exact mismatch risks before you ship.
