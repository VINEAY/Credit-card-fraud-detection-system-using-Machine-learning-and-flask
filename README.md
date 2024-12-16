# Credit-card-fraud-detection-system-using-Machine-learning-and-flask
**Credit Card Fraud Detection System**
An intelligent system utilizing machine learning to identify fraudulent credit card transactions. Features a Flask-based web interface and an XGBoost model with 96.59% accuracy. Designed for real-time fraud detection with confidence scores, providing actionable insights for financial institutions.
# Credit Card Fraud Detection System

**Developed By:**
1. **Keerthana Aluri**


---

## Abstract:
This project demonstrates an intelligent fraud detection system leveraging machine learning to identify fraudulent credit card transactions. Key components include:

- A **Flask-based web interface** for user-friendly interaction.
- An **XGBoost machine learning model** achieving 96.59% accuracy in fraud detection.
- Real-time transaction analysis delivering actionable insights with confidence scores for fraud mitigation.

---

## Table of Contents:
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Future Enhancements](#future-enhancements)
5. [Conclusion](#conclusion)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
8. [References](#references)

---

## Introduction
### 1.1 Problem Statement:
Fraudulent credit card transactions pose a significant challenge to financial institutions, leading to substantial financial and reputational losses. Detecting these transactions is challenging due to their rarity and the diversity of legitimate user behaviors.

### 1.2 Objectives:
- Build a machine learning model to classify transactions as fraudulent or legitimate.
- Develop a web-based interface for real-time transaction analysis.
- Provide actionable insights with fraud risk scoring.

---

## Methodology
### 2.1 Data Preprocessing:
- **Data Cleaning:**
  - Removed irrelevant columns and rows with missing values.
  - Combined datasets, resulting in 1,852,394 samples (9,651 fraudulent).
- **Class Balancing:**
  - Balanced fraud and non-fraud cases using undersampling.
- **Feature Engineering:**
  - Extracted temporal and geographic features.
  - Encoded categorical variables and scaled numerical data.
- **Train-Test Split:**
  - Split the data (80% training, 20% testing) using stratified sampling.

### 2.2 Model Development:
- **XGBoost:** Achieved 96.59% testing accuracy with robust generalization.
- Compared with Logistic Regression and Random Forest for performance benchmarking.

---

## Results
### 3.1 Performance Metrics:
- **XGBoost:**
  - Testing Accuracy: 96.59%
  - Precision: 97.15%, Recall: 95.98%, F1 Score: 96.57%
- **Random Forest:**
  - Testing Accuracy: 95.73%
- **Logistic Regression:**
  - Testing Accuracy: 85.13%

### 3.2 Visualizations:
- Performance metrics comparison via bar charts.
- Confusion matrices for detailed analysis.

---

## Future Enhancements:
1. **Real-Time Deployment:** Transition to production for real-time fraud detection.
2. **Deep Learning Models:** Explore neural network architectures for higher accuracy.
3. **Mobile Integration:** Extend capabilities to mobile devices.
4. **Additional Data Sources:** Incorporate transaction history and social network analysis.

---

## Conclusion:
The Credit Card Fraud Detection System effectively applies machine learning for robust fraud detection. Its high accuracy, usability, and scalability position it as a cornerstone tool for financial fraud prevention. With future enhancements, the system can adapt to evolving threats and provide even greater value.

---

## Getting Started:
### Prerequisites:
- Python 3.8 or above
- Flask
- XGBoost
- Pandas, NumPy, Scikit-learn, Matplotlib

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application:
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Usage:
- Upload transaction data via the web interface.
- View real-time fraud risk assessments and actionable recommendations.

---

## References:
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- Dataset Source: Confidential (project-specific data)

---

> **Note:** For detailed methodology and experimental results, refer to the `Technical_Project_Report.pdf` included in this repository.

