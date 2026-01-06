# Advanced Time Series Forecasting with Deep Learning and Explainability

## Project Overview

This project implements an end-to-end, production-style time series forecasting system using deep learning models. The focus is on predicting stock closing prices from multivariate historical data while ensuring robust evaluation and interpretability. Two advanced architectures are explored: **Long Short-Term Memory (LSTM)** networks and **Transformer-based models**. In addition to predictive accuracy, the project integrates explainability using **SHAP (SHapley Additive exPlanations)** to understand feature contributions across time.

The project is designed to meet academic and industry standards, emphasizing clean code, reproducibility, walk-forward validation, and transparent model interpretation.

---

## Dataset

* **Source**: NASDAQ historical stock market data
* **Company**: Apple Inc. (AAPL)
* **Format**: CSV
* **Frequency**: Daily
* **Features**:

  * Open
  * High
  * Low
  * Close
  * Volume
* **Target Variable**: Close price

The `Date` column is used as the temporal index. All numerical features are scaled using Min-Max normalization to support stable neural network training.

---

## Project Structure

```
├── data/
│   └── AAPL.csv
├── notebooks/
│   ├── lstm_model.ipynb
│   ├── transformer_model.ipynb
│   └── shap_analysis.ipynb
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.csv
│   └── prediction_plots.png
├── README.md
└── requirements.txt
```

---

## Methodology

### Data Preprocessing

1. Load and sort data chronologically
2. Handle missing values
3. Scale features using MinMaxScaler
4. Convert the time series into supervised learning sequences using a sliding window

A fixed look-back window of past observations is used to predict the next closing price.

---

## Models Implemented

### 1. LSTM Model

The LSTM model is designed to capture long-term temporal dependencies in financial time series.

**Architecture Details**:

* Input: Multivariate sequences
* LSTM layers: 2
* Hidden units: 64
* Dropout: 0.2
* Optimizer: Adam
* Learning rate: 0.001
* Loss function: Mean Squared Error (MSE)

LSTM models perform well in stable market conditions by leveraging recent price trends.

---

### 2. Transformer Model

The Transformer model uses self-attention mechanisms to capture global dependencies across time steps.

**Architecture Details**:

* Input projection layer
* Transformer encoder layers: 2
* Attention heads: 4
* Feedforward dimension: 128
* Dropout: 0.1
* Optimizer: Adam
* Learning rate: 0.001

Transformers demonstrate stronger performance during volatile market periods due to their ability to model long-range interactions.

---

## Training Strategy

* Batch size: 32
* Epochs: 10–20 (depending on experiment)
* Time-aware train-test split (80/20)
* No data shuffling to preserve temporal order

---

## Walk-Forward Validation

To simulate real-world forecasting scenarios, walk-forward validation is used. The model is trained on an expanding window of historical data and evaluated on the subsequent period. This process is repeated multiple times, and error metrics are averaged.

This approach prevents data leakage and provides a realistic estimate of out-of-sample performance.

---

## Evaluation Metrics

The following metrics are used for quantitative evaluation:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

Both LSTM and Transformer models are compared against a naive baseline model.

---

## Model Explainability (SHAP)

SHAP is applied to interpret the predictions of deep learning models. It quantifies the contribution of each feature at each time step toward the final forecast.

Explainability analysis is performed for three distinct periods:

1. Stable market conditions
2. High volatility phase
3. Trend reversal or recovery period

SHAP visualizations (summary plots and force plots) help identify how price levels and trading volume influence predictions under different market regimes.

---

## Results

* Deep learning models outperform the naive baseline in both MAE and RMSE
* Transformer models show improved robustness during volatile periods
* SHAP analysis confirms that recent closing prices dominate predictions, while volume and price range become more influential during high volatility

---

## Conclusion

This project demonstrates that advanced deep learning architectures can significantly improve multivariate time series forecasting performance. By integrating explainability and walk-forward validation, the system ensures both accuracy and transparency, making it suitable for real-world financial forecasting applications.

---

## Future Enhancements

* Multi-step forecasting
* Hyperparameter optimization
* Inclusion of technical indicators
* Deployment as a real-time forecasting service

---

## Requirements

* Python 3.9+
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SHAP

---


