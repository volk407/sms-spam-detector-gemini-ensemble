# 📡 SMS Spam Detector + Gemini (Ensemble AI Classifier)

A modern SMS/email spam detector powered by:
✅ Classical ML (TF-IDF + Logistic Regression + Isotonic Calibration)  
✅ Optional Gemini (2.0 Flash) reasoning  
✅ Phishing heuristic scoring  
✅ Clean Gradio interface  

---

## ✅ Team
**👤 Name: Moustafa Ahmed Ismail**

📧 Contact: m.ageaaismn@yahoo.com

🎓 Electrical & Computer Engineering – Constructor University Bremen  

💻 Focus Areas: Cybersecurity, Risk Engineering, Applied AI and User Interface Design

**👤 Name: Sky**  
📧 Contact: mashoguliashvili00@gmail.com

🎓 Electrical & Computer Engineering – Constructor University Bremen

💻 Focus Areas: Software Engineering, Machine Learning, and User Interface Design

---

## 🚀 Features
- ✅ Trains a spam classifier from `spam.csv`
- ✅ Handles multiple dataset formats (Kaggle, custom)
- ✅ Probability-based prediction
- ✅ Optional Gemini API judge (JSON-strict)
- ✅ UI shows model %, Gemini %, rationale, and final verdict
- ✅ auto-clears proxies to avoid API blocking

---

## 📁 Repository Structure

sms-spam-detector-gemini-ensemble/
│
├── main.py # Gradio UI + Gemini integration
├── train_model.py # Training script (run once)
├── model/spam_calibrated.joblib
├── requirements.txt
├── .gitignore
└── README.md


---

## 🔧 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detector-gemini-ensemble.git
cd sms-spam-detector-gemini-ensemble
pip install -r requirements.txt
```

---

## 📊 2. Prepare Dataset

Place your dataset at:

data/spam.csv

Accepted formats include Kaggle’s SMS Spam Collection v1/v2.
The script auto-detects label/text columns.

---

## 🧠 3. Train Model

python train_model.py

This will output:

model/spam_calibrated.joblib

## 🖥 4. Run App

python main.py

Open the Gradio URL in the browser and test messages.
#🔑 (Optional) Enable Gemini

Inside the UI:

    Tick ✅ “Use Gemini”

    Paste your Gemini API Key

Gemini outputs:

    spam %

    legit %

    reasoning

    combined final verdict (70% Gemini + 30% model)

✅ Example Output (Final verdict)

Model spam: 87%
Gemini spam: 92%
Final result: SPAM (90.1%)

## 🧠 Tech Behind It
Component	Purpose
TF-IDF (1–2 grams)	Robust lexical spam features
Logistic Regression	High-precision binary classifier
Isotonic Calibration	Probability reliability
Gemini Judge	Language-aware semantic validation
Heuristic Scoring	Detects phishing tricks & URLs
