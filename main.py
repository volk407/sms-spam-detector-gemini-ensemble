import os
for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
          "GOOGLE_API_BASE_URL","GOOGLE_API_BASE","API_BASE"]:
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"
print("✅ Environment cleaned — Gemini will connect directly.")


# --- Setup & installs ---
import sys, subprocess, importlib.util

def ensure(package):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

for p in ["pandas", "scikit-learn", "numpy", "joblib", "matplotlib", "gradio", "google-generativeai", "scipy"]:
    ensure(p)

import pandas as pd
import numpy as np
import os, re, json, time, html

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

print("✅ Environment ready!")

# --- Load/Upload spam.csv ---
csv_candidates = ["spam.csv", "/content/spam.csv", "/content/drive/MyDrive/spam.csv"]
csv_path = None
for c in csv_candidates:
    if os.path.exists(c):
        csv_path = c
        break

if csv_path is None:
    try:
        from google.colab import files
        print("Please upload your spam.csv file ⬆️")
        uploaded = files.upload()
        for k in uploaded.keys():
            if k.lower().endswith(".csv"):
                csv_path = k
                break
    except Exception:
        print("Colab upload UI not available here. Make sure spam.csv is in the working directory.")
        if os.path.exists("spam.csv"):
            csv_path = "spam.csv"

assert csv_path is not None, "Could not find spam.csv. Upload it and rerun."
print("Using CSV:", csv_path)

# --- Read & normalize dataset ---
df_raw = pd.read_csv(csv_path, encoding="latin-1")
print("Columns detected:", list(df_raw.columns))

# auto-detect label/text columns (handles Kaggle v1/v2)
cands_label = ["label", "Label", "category", "Category", "v1", "class", "Class"]
cands_text  = ["text", "Text", "message", "Message", "sms", "SMS", "v2"]

label_col = next((c for c in cands_label if c in df_raw.columns), None)
text_col  = next((c for c in cands_text  if c in df_raw.columns), None)
if label_col is None or text_col is None:
    if len(df_raw.columns) >= 2:
        label_col = df_raw.columns[0]
        text_col  = df_raw.columns[1]
    else:
        raise ValueError("Couldn't auto-detect label/text columns. Please rename to 'label' and 'text'.")

df = df_raw[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
df["text"] = df["text"].astype(str).str.strip()
df = df.dropna(subset=["label", "text"]).drop_duplicates(subset=["text"])

# Map labels (phishing considered spam)
label_map = {"ham":0,"Ham":0,"HAM":0,"not spam":0,"legit":0,"normal":0,
             "spam":1,"Spam":1,"SPAM":1,"junk":1,"phishing":1}
def normalize_label(x):
    if x in label_map:
        return label_map[x]
    try:
        return int(x)
    except:
        return 1 if str(x).lower().strip() in ["spam","junk","phishing"] else 0

df["y"] = df["label"].apply(normalize_label).astype(int)
print("Label counts:", df["y"].value_counts().to_dict())
print(df.head())

# --- Train/validation split ---
X = df["text"].values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# =========================
#   ALGORITHM SWITCH (your requested baseline)
#   TF-IDF (1–2 grams) -> LogisticRegression (liblinear) -> Isotonic Calibration
# =========================

# Compute class weights for imbalance
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {cls: w for cls, w in zip(classes, cw)}
print("Class weights:", class_weight)

# Vectorizer + Base classifier
base_clf = LogisticRegression(
    max_iter=2000,
    class_weight=class_weight,
    solver="liblinear"
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        min_df=2,
        strip_accents="unicode"
    )),
    ("clf", base_clf),
])

# Fit baseline pipeline
pipe.fit(X_train, y_train)

# Calibrate probabilities (isotonic)
calibrated = CalibratedClassifierCV(pipe, cv=5, method="isotonic")
calibrated.fit(X_train, y_train)

print("✅ Model trained & calibrated!")

# --- Evaluate ---
y_proba = calibrated.predict_proba(X_test)[:,1]
y_pred  = (y_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
try:
    print("ROC-AUC :", roc_auc_score(y_test, y_proba))
except Exception as e:
    print("ROC-AUC not available:", e)
print()
print(classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model artifact ---
os.makedirs("model", exist_ok=True)
joblib.dump(calibrated, "model/spam_calibrated.joblib")
print("Saved: model/spam_calibrated.joblib")

# =========================
#   Transparent Phishing Heuristic (kept internal; UI unchanged)
# =========================
PHISH_KEYWORDS = [
    "verify","verification","account","password","passcode","otp","one-time","code",
    "login","log in","sign in","reset","confirm","update","security",
    "suspend","suspension","locked","unlock","unusual","activity",
    "urgent","immediately","24 hours","limited time","act now",
    "bank","paypal","apple id","amazon","microsoft","netflix","invoice","package",
    "gift card","voucher","prize","winner"
]
SHORTENERS = ["bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","t.ly"]
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def phishing_heuristic(text: str) -> float:
    t = text.lower()
    score = 0.0
    urls = URL_PATTERN.findall(text)
    if urls:
        score += 1.5
        if any(s in t for s in SHORTENERS):
            score += 1.0
    hits = sum(1 for kw in PHISH_KEYWORDS if kw in t)
    score += 0.5 * hits
    if "click" in t or "tap" in t:
        score += 0.5
    if any(ext in t for ext in [".html",".php",".exe",".apk"]):
        score += 0.5
    prob = 1 / (1 + np.exp(-(score - 2.0)))
    return float(np.clip(prob, 0.0, 1.0))

# --- Predictor (returns same keys your UI expects; heuristic is extra) ---
CAL_MODEL = joblib.load("model/spam_calibrated.joblib")

def predict_message(msg: str):
    spam_prob = float(CAL_MODEL.predict_proba([msg])[0,1])  # 0..1
    final_label = "SPAM" if spam_prob >= 0.5 else "HAM"
    # heuristic computed but not required by your UI; keeping it available
    _phish = phishing_heuristic(msg)
    return {
        "label": final_label,
        "spam_probability_pct": round(spam_prob * 100.0, 2),
        "phishing_likelihood_pct": round(_phish * 100.0, 2),
    }

# Sanity check
print(predict_message("URGENT: Your account has been suspended. Verify at http://bit.ly/fake"))

# =========================
#   GEMINI (Spam / Legit only, fast, robust)
#   (UNCHANGED AS REQUESTED)
# =========================
import google.generativeai as genai
import requests

GEMINI_MODEL_NAME = "gemini-2.0-flash"  # fast
GEMINI_SYSTEM_PROMPT = """You are a spam detector.
Return ONLY JSON in this exact form and make sure percentages sum to 100 (after rounding to 2 decimals):
{
  "spam_pct": 0.0,
  "legit_pct": 0.0,
  "rationale": "very short reason"
}
No extra text outside JSON.
"""

def _clear_env():
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy",
              "GOOGLE_API_BASE_URL","GOOGLE_API_BASE","API_BASE"]:
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"

def _safe_json(txt: str) -> dict:
    if not txt:
        raise ValueError("Empty text")
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Gemini returned non-JSON / empty")

def _extract_text_from_rest(body: dict) -> str:
    try:
        cands = body.get("candidates", []) or []
        for c in cands:
            parts = (c.get("content") or {}).get("parts", []) or []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    return p["text"]
    except Exception:
        pass
    return ""

def _normalize_two_way(data: dict) -> dict:
    spam = float(data.get("spam_pct", 0.0) or 0.0)
    legit = float(data.get("legit_pct", 0.0) or 0.0)
    s = spam + legit
    if s <= 0:
        out = {"spam_pct": 0.0, "legit_pct": 100.0}
    else:
        out = {"spam_pct": round(100.0 * spam / s, 2), "legit_pct": round(100.0 * legit / s, 2)}
    out["rationale"] = (data.get("rationale") or "").strip() or "(no rationale)"
    return out

def gemini_judge(message: str, api_key: str, timeout_s: int = 3) -> dict:
    if not api_key:
        return {"spam_pct":0.0,"legit_pct":100.0,"rationale":"Gemini skipped"}

    _clear_env()
    # Try SDK
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=GEMINI_SYSTEM_PROMPT,
            generation_config={"temperature": 0, "top_p": 1, "response_mime_type": "application/json"},
        )
        prompt = f'Message:\n"""{message}"""'
        resp = model.generate_content(prompt, request_options={"timeout": timeout_s})

        txt = getattr(resp, "text", "") or ""
        if not txt:
            try:
                txt = _extract_text_from_rest(resp.to_dict())
            except Exception:
                txt = ""
        if not txt:
            return {"spam_pct": 50.0, "legit_pct": 50.0,
                    "rationale": "Gemini returned no text (safety or empty response)."}

        data = _safe_json(txt)
        return _normalize_two_way(data)

    except Exception:
        # REST fallback
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
            payload = {
                "systemInstruction": {"parts":[{"text":GEMINI_SYSTEM_PROMPT}]},
                "generationConfig":{"temperature":0, "topP":1, "responseMimeType":"application/json"},
                "contents":[{"parts":[{"text": f'Message:\n"""{message}"""'}]}]
            }
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            body = r.json()
            txt = _extract_text_from_rest(body)
            if not txt:
                return {"spam_pct": 50.0, "legit_pct": 50.0,
                        "rationale": "Gemini REST returned no text (safety or empty response)."}

            data = _safe_json(txt)
            return _normalize_two_way(data)

        except Exception:
            return {"spam_pct": 50.0, "legit_pct": 50.0,
                    "rationale": "Gemini call failed; returned neutral result."}

# --- Final result logic ---
# Gemini OFF  -> final = model %
# Gemini ON   -> final = 0.3 * model + 0.7 * gemini  (Gemini-biased)
def final_result(spam_model: float, spam_gemini: float, gemini_used: bool) -> str:
    if not gemini_used:
        final_spam = round(spam_model, 2)
    else:
        final_spam = round(0.3 * float(spam_model) + 0.7 * float(spam_gemini), 2)

    if final_spam >= 50.0:
        return f"Final result: SPAM ({final_spam}%)"
    else:
        return f"Final result: HAM ({round(100.0 - final_spam, 2)}%)"

# --- Pretty blocks for Gemini ---
def gemini_percent_block(spam_pct: float, legit_pct: float) -> str:
    return f"""
    <div style="border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa;">
      <div style="font-weight:600; margin-bottom:6px;">Gemini percentages</div>
      <div>spam = <strong>{spam_pct}%</strong> &nbsp;|&nbsp; legit = <strong>{legit_pct}%</strong></div>
    </div>
    """

def gemini_flags_block(rationale: str) -> str:
    text = rationale.replace(" and ", ", ")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        parts = [rationale.strip()]
    lis = "".join([f'<li style="margin:6px 0;">🚩 {html.escape(p)}</li>' for p in parts])
    return f"""
    <div style="border:1px solid #fecaca; border-radius:12px; padding:12px; background:#fff1f2;">
      <div style="font-weight:600; color:#991b1b; margin-bottom:6px;">Red flags</div>
      <ul style="margin:0 0 0 16px; padding:0;">{lis}</ul>
    </div>
    """

# --- Gradio app (UNCHANGED) ---
import gradio as gr

def gradio_predict(msg, use_gemini, gemini_api_key):
    out = predict_message(msg)
    spam_model = out["spam_probability_pct"]

    gemini_used = False
    gem_block = '<div style="opacity:0.6;">Gemini not used.</div>'
    gem_flags = ""

    spam_gemini = 0.0
    if use_gemini and (gemini_api_key or "").strip():
        gemini_used = True
        data = gemini_judge(msg, gemini_api_key.strip(), timeout_s=3)
        spam_gemini = data["spam_pct"]
        gem_block = gemini_percent_block(data["spam_pct"], data["legit_pct"])
        gem_flags = gemini_flags_block(data["rationale"]) if data["spam_pct"] >= 50.0 else ""

    final = final_result(spam_model, spam_gemini, gemini_used)
    return out["label"], spam_model, gem_block, gem_flags, final

with gr.Blocks(title="Spam Detector + Gemini") as demo:
    gr.Markdown("# SMS Spam Detector (+ Optional Gemini)")

    msg = gr.Textbox(label="Enter message", lines=3, placeholder="Paste an SMS/email…")
    with gr.Row():
        use_gemini = gr.Checkbox(label="Use Gemini", value=False)
    with gr.Accordion("Gemini API key", open=False):
        gem_key = gr.Textbox(label="API Key", type="password", placeholder="Paste your Gemini API key")

    btn = gr.Button("Analyze")

    pred_label   = gr.Label(label="Model Prediction (HAM/SPAM)")
    spam_pct_box = gr.Number(label="Model spam probability (%)", precision=2)

    gr.Markdown("### Gemini opinion")
    gem_percent  = gr.HTML()
    gem_rationale = gr.HTML()

    gr.Markdown("### ✅ Final result")
    final_text   = gr.Label(label="Verdict")

    btn.click(
        fn=gradio_predict,
        inputs=[msg, use_gemini, gem_key],
        outputs=[pred_label, spam_pct_box, gem_percent, gem_rationale, final_text]
    )

demo.launch(share=False)
