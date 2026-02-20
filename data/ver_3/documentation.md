# SafeX Android (Emulator) Integration — TFLite Char-CNN (Binary Scam Detector)

This doc shows **exactly** how to integrate the trained **TFLite** model into your Android app (emulator + VS Code workflow). It includes **all required files**, **where to put them**, **all Kotlin code**, and **how to run + validate**.

---

## 0) What you should already have

From Kaggle `/kaggle/working/` (or your shared drive), you should have:

* `safex_charcnn_dynamic.tflite` *(recommended default)*
* `char_vocab.json`
* `model_config.json`
* (optional but recommended) `demo_examples.json` *(10–100 labeled examples for QA)*

These are the only artifacts needed to run inference on-device.

---

## 1) Put model files into your Android project

Create this folder structure:

```
app/src/main/assets/models/
  safex_charcnn_dynamic.tflite
  char_vocab.json
  model_config.json
  demo_examples.json   (optional)
```

> Tip: Keep everything inside `assets/models/` so teammates can find it quickly.

---

## 2) Add dependencies (Gradle)

In `app/build.gradle` (Module: app), add:

```gradle
dependencies {
    implementation "org.tensorflow:tensorflow-lite:2.16.1"

    // Optional helpers (OK to include)
    implementation "org.tensorflow:tensorflow-lite-support:0.4.4"
}
```

Sync Gradle.

---

## 3) Preprocessing spec (MUST match training)

Your model is **character-level** and expects **int32 IDs**.

### 3.1 Normalization rules

Apply these rules before tokenizing:

1. **Unicode NFKC** normalization
2. Collapse whitespace: `\s+ -> " "` and `.trim()`
3. Replace URL-like tokens with `<URL>`

   * pattern: `(?i)\b(?:https?://|www\.)\S+`
4. Replace 8+ consecutive digits with `<NUM>`

   * pattern: `\b\d{8,}\b`

### 3.2 Tokenization rules

1. Split into **characters** (`toCharArray()`)
2. Truncate to `SEQ_LEN` (from `model_config.json`)
3. Map each char to its vocab index:

   * `pad_index = 0`
   * `unk_index = 1`
4. Pad to `SEQ_LEN` using `pad_index`

### 3.3 Model IO

* **Input**: `int32[1, SEQ_LEN]`
* **Output**: `float32[1, 1]` (scam probability)
* Default decision: `score >= threshold` (from `model_config.json`)

---

## 4) Drop-in Kotlin integration (ScamDetector)

Create file:

`app/src/main/java/<your/package>/ml/ScamDetector.kt`

```kotlin
package your.pkg.ml

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.text.Normalizer

data class ModelConfig(
    val seqLen: Int,
    val threshold: Double,
    val padIndex: Int,
    val unkIndex: Int,
)

class ScamDetector(ctx: Context) {

    private val cfg: ModelConfig
    private val vocab: Map<String, Int>
    private val tflite: Interpreter

    init {
        // ---- load config ----
        val cfgJson = ctx.assets.open("models/model_config.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val cfgObj = JSONObject(cfgJson)
        cfg = ModelConfig(
            seqLen = cfgObj.getInt("seq_len"),
            threshold = cfgObj.getDouble("threshold"),
            padIndex = cfgObj.getInt("pad_index"),
            unkIndex = cfgObj.getInt("unk_index"),
        )

        // ---- load vocab ----
        val vocabJson = ctx.assets.open("models/char_vocab.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        vocab = loadVocab(vocabJson)

        // ---- load TFLite model bytes ----
        val modelBytes = ctx.assets.open("models/safex_charcnn_dynamic.tflite").readBytes()

        // ---- Interpreter options (latency) ----
        val opts = Interpreter.Options().apply {
            setNumThreads(4) // adjust: 2-6; test on device
        }

        tflite = Interpreter(modelBytes, opts)
    }

    private fun loadVocab(jsonText: String): Map<String, Int> {
        val arr = JSONArray(jsonText)
        val map = HashMap<String, Int>(arr.length())
        for (i in 0 until arr.length()) {
            map[arr.getString(i)] = i
        }
        return map
    }

    private fun normalizeForModel(s: String): String {
        var x = Normalizer.normalize(s, Normalizer.Form.NFKC)
        x = x.replace(Regex("\\s+"), " ").trim()
        x = x.replace(Regex("(?i)\\b(?:https?://|www\\.)\\S+"), "<URL>")
        x = x.replace(Regex("\\b\\d{8,}\\b"), "<NUM>")
        return x
    }

    private fun toCharIds(text: String): IntArray {
        val x = normalizeForModel(text)
        val ids = IntArray(cfg.seqLen) { cfg.padIndex }

        val chars = x.toCharArray()
        val n = minOf(chars.size, cfg.seqLen)
        for (i in 0 until n) {
            val ch = chars[i].toString()
            ids[i] = vocab[ch] ?: cfg.unkIndex
        }
        return ids
    }

    /** scam probability 0..1 */
    fun predictScore(text: String): Float {
        val ids = toCharIds(text)
        val input = arrayOf(ids)                 // shape [1, seqLen]
        val output = Array(1) { FloatArray(1) }  // shape [1, 1]
        tflite.run(input, output)
        return output[0][0]
    }

    fun predictIsScam(text: String): Boolean {
        return predictScore(text) >= cfg.threshold.toFloat()
    }

    fun threshold(): Double = cfg.threshold
    fun seqLen(): Int = cfg.seqLen
}
```

---

## 5) Quick emulator UI (manual testing)

Add a simple screen with an `EditText`, `Button`, and `TextView`.

Example in an Activity (core logic):

```kotlin
val detector = ScamDetector(this)

btnDetect.setOnClickListener {
    val msg = editText.text.toString()

    val t0 = System.nanoTime()
    val score = detector.predictScore(msg)
    val t1 = System.nanoTime()

    val isScam = score >= detector.threshold().toFloat()
    val ms = (t1 - t0) / 1_000_000.0

    resultText.text = "score=%.4f  label=%s  time=%.2fms"
        .format(score, if (isScam) "SCAM" else "BENIGN", ms)
}
```

---

## 6) Optional: automated QA using `demo_examples.json`

Create `demo_examples.json` in `assets/models/`:

```json
[
  {"text": "Maybank: Your TAC is <OTP>. Do not share.", "expected": 0},
  {"text": "Verify now at <URL> to unblock account. TAC <OTP>", "expected": 1}
]
```

Then add a debug-only test runner (e.g., on a hidden button):

```kotlin
import org.json.JSONArray

fun runDemoExamples(ctx: Context, detector: ScamDetector): Pair<Int, Int> {
    val json = ctx.assets.open("models/demo_examples.json")
        .bufferedReader(Charsets.UTF_8)
        .use { it.readText() }
    val arr = JSONArray(json)

    var ok = 0
    for (i in 0 until arr.length()) {
        val o = arr.getJSONObject(i)
        val text = o.getString("text")
        val expected = o.getInt("expected")
        val pred = if (detector.predictIsScam(text)) 1 else 0
        if (pred == expected) ok++
    }
    return ok to arr.length()
}
```

---

## 7) Run on emulator from VS Code

You can **code in VS Code**, but you still need:

* Android SDK + `adb`
* An emulator (usually easiest to launch via Android Studio Device Manager)

### 7.1 Build + install

From repo root:

```bash
./gradlew installDebug
```

### 7.2 Confirm emulator is connected

```bash
adb devices
```

### 7.3 View logs

```bash
adb logcat
```

---

## 8) What to commit to GitHub (so teammate can plug-and-play)

Commit these paths:

### 8.1 Model assets

* `app/src/main/assets/models/safex_charcnn_dynamic.tflite`
* `app/src/main/assets/models/char_vocab.json`
* `app/src/main/assets/models/model_config.json`
* `app/src/main/assets/models/demo_examples.json` *(optional but recommended)*

### 8.2 Code

* `app/src/main/java/<pkg>/ml/ScamDetector.kt`
* The demo Activity/Fragment UI that calls it

### 8.3 Docs

* This file: `documentation.md`

### 8.4 Git LFS (only if your TFLite is big)

If `*.tflite` is too large for normal git, enable Git LFS:

```bash
git lfs install
git lfs track "*.tflite"
git add .gitattributes
git add app/src/main/assets/models/safex_charcnn_dynamic.tflite
```

---

## 9) Common issues & fixes

### 9.1 Wrong predictions vs Kaggle

Almost always preprocessing mismatch.

* Ensure you apply **NFKC**, whitespace collapse, URL→`<URL>`, 8+ digits→`<NUM>`
* Ensure `seq_len` matches `model_config.json`
* Ensure `pad_index=0`, `unk_index=1`

### 9.2 Slow inference

* Set `Interpreter.Options().setNumThreads(2..6)` and benchmark
* Prefer `dynamic` on CPU; try `fp16` if you plan to use GPU delegate

### 9.3 ZH/MIX looks bad

* Confirm your vocab JSON loaded properly
* Confirm you’re not accidentally doing lowercase / stripping non-ASCII

---

## 10) Recommended next checklist (before demo)

1. Test 10 mixed EN/MS/ZH/MIX messages in emulator and log score+label
2. Run `demo_examples.json` QA → expect high pass rate
3. Measure latency on emulator (and ideally one physical phone)
4. Decide threshold (default 0.35) and optionally expose “Strict/Balanced/Aggressive” modes

---

## Appendix: File expectations

* `char_vocab.json` is an array. Index is token id.

  * index 0 = padding
  * index 1 = [UNK]
* Model input must be `int32`.
* Output is a single float probability.
