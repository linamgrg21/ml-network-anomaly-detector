# ML Network Anomaly Detector

Machine learning-based anomaly detection system for identifying suspicious network traffic in wireless and cloud environments.

## Features

- Network intrusion detection using Random Forest
- NSL-KDD dataset support
- Automatic preprocessing and encoding
- Classification report generation
- Confusion matrix visualization
- Model export using Joblib
- NSL-KDD Dataset:
https://www.kaggle.com/datasets/hassan06/nslkdd


## Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Project Structure

```text
ml-network-anomaly-detector/
│
├── data/
├── results/
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training script:

```bash
python src/train.py
```

## Output

The project generates:
- Trained ML model
- Classification report
- Confusion matrix image

## Author

Linamgrg21


