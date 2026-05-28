# ml-network-anomaly-detector

Machine learning-based anomaly detection system for identifying suspicious network traffic in wireless and cloud environments.

---

# Network Anomaly Detection Using Machine Learning

## Overview

This project implements a machine learning-based network anomaly detection system using the NSL-KDD dataset. The model identifies malicious or suspicious network traffic patterns and classifies potential cyberattacks.

The system was trained using the `KDDTrain+` dataset and evaluated using the separate `KDDTest+` dataset to provide a more realistic and unbiased performance evaluation.

---

## Features

- Machine learning intrusion detection
- Network traffic anomaly classification
- Random Forest model implementation
- Data preprocessing and feature engineering
- Separate train/test dataset evaluation
- Classification performance reporting
- Confusion matrix visualization
- Trained model export using Joblib

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset

NSL-KDD Dataset:

https://www.kaggle.com/datasets/hassan06/nslkdd

---

## Project Structure

```text
ml-network-anomaly-detector/
├── app/
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── README.md
├── notebooks/
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── model.pkl
│   └── README.md
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

---

## How To Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the training script

```bash
python src/train.py
```

---

## Model Used

### Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Why Random Forest?

- Handles large datasets efficiently
- Works well with classification problems
- Reduces overfitting compared to single decision trees
- Provides strong baseline performance for intrusion detection systems

---

## Results

The Random Forest model was trained on `KDDTrain+` and evaluated on the separate `KDDTest+` dataset.

| Class | Precision | Recall | F1-Score |
|------|------|------|------|
| Normal Traffic | 0.65 | 0.97 | 0.78 |
| Attack Traffic | 0.97 | 0.61 | 0.75 |

### Overall Accuracy

```text
77%
```

The use of a separate test dataset provides a more realistic evaluation compared to splitting the same dataset for training and testing.

---

## Output Files

The project generates:

- Trained machine learning model (`model.pkl`)
- Classification report (`classification_report.txt`)
- Confusion matrix visualization (`confusion_matrix.png`)

---

## Future Improvements

- XGBoost implementation
- Deep learning-based anomaly detection
- Real-time traffic monitoring
- Flask dashboard integration
- SIEM integration
- Live packet capture using Scapy
- Cloud-based deployment

---

## Author

Linamgrg21