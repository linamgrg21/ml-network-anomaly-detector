# ml-network-anomaly-detector
Machine learning-based anomaly detection system for identifying suspicious network traffic in wireless and cloud environments.

Network Anomaly Detection Using Machine Learning
Overview

This project implements a machine learning-based network anomaly detection system using the NSL-KDD dataset. The model identifies malicious or suspicious network traffic patterns and classifies potential attacks.

Features
Machine learning intrusion detection
Network traffic anomaly classification
Random Forest model implementation
Data preprocessing and feature engineering
Classification performance reporting
Technologies Used
Python
Pandas
Scikit-learn
NumPy
Matplotlib
Jupyter Notebook
Dataset

NSL-KDD Dataset:
https://www.kaggle.com/datasets/hassan06/nslkdd

Project Structure
ml-network-anomaly-detector/
├── app/
├── data/
├── notebooks/
├── results/
├── src/
│   └── train.py
├── requirements.txt
└── README.md

How TO Run
pip install -r requirements.txt
python src/train.py


Sample Output
Accuracy: 1.0

Future Improvements
Real-time traffic monitoring
Flask dashboard integration
Deep learning implementation
SIEM integration
Live packet capture using Scapy