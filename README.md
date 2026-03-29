# 📧 Spam Mail Detection

## 🔹 Overview
This project builds a **machine learning-based spam detection system** using NLP techniques and classification models. The primary focus is on **real-world applicability**, where incorrectly marking important emails as spam (false positives) must be minimized.

---

## 🎯 Problem Statement
Design a spam classifier that:
- Detects spam emails effectively
- Minimizes **false positives** (critical business requirement)
- Maintains strong generalization on unseen data

---

## ⚙️ Approach

### 🔸 Feature Engineering
- **TF-IDF Vectorization**
  - max_features = 5000
  - ngram_range = (1,2)
  - stop_words = 'english'
- **Additional Feature**
  - log_length (log-transformed email length)

👉 Final feature space = Text features + structural feature

---

### 🔸 Model Exploration
The following models were evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- XGBoost
- LightGBM

Hyperparameter tuning was performed using **cross-validation**.

---

### 🔸 Model Selection Strategy
- Focused on **precision-first optimization**
- Selected model based on:
  - High precision
  - Low false positives
  - Stable recall

👉 **Logistic Regression** performed best overall

---

### 🔸 Threshold Tuning
- Default threshold (0.5) was not optimal
- Threshold optimized to **0.7**
- Result:
  - Reduced false positives significantly
  - Maintained good recall

---

## ⚠️ Pipeline Correction

### Initial Limitation
- TF-IDF was fit on full dataset
- Evaluation was done on same data

👉 This introduced **data leakage**

---

### Final Pipeline (Corrected)
1. Train-Test Split (80/20)
2. TF-IDF fit only on training data
3. Cross-validation on training set
4. Final evaluation on test set

---

## 📊 Final Results (Test Set)

### Classification Report
- Accuracy: **98%**
- Precision (Spam): **0.98**
- Recall (Spam): **0.83**
- F1 Score: **0.90**

---

### Confusion Matrix

[[901    2][22    109]]

- False Positives: **2**
- False Negatives: **22**

---

## 💼 Business Interpretation
- Only **2 out of 1034 emails (~0.2%)** were incorrectly marked as spam
- Extremely low risk of losing important emails
- Some spam is missed, but this is acceptable in real-world systems

👉 Model is optimized for **high precision and reliability**

---

## 📈 ROC Curve
- AUC Score: **0.9911**
- Indicates excellent class separation
- Confirms strong generalization on unseen data

---

## 🧠 Key Learnings
- TF-IDF effectively captures text importance
- log transformation helps handle skewed distributions
- Threshold tuning is critical for business alignment
- Cross-validation is useful for model selection
- Train-test split is essential to avoid data leakage

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- XGBoost
- LightGBM

---

## 🚀 Output Artifacts
- `spam_model.pkl` → trained model
- `tfidf.pkl` → vectorizer
- `threshold.txt` → decision threshold
- `predictions.csv` → model predictions

---

## ✅ Conclusion
The final model achieves a strong balance between **accuracy and business constraints**, ensuring minimal false positives while maintaining effective spam detection. The pipeline is designed to be **production-ready and scalable**.

---

## 📌 Future Improvements
- Deploy using Streamlit or Flask
- Add real-time email classification API
- Use deep learning models (LSTM/BERT)
- Improve recall without increasing false positives
