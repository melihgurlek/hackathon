# Ford DTC Predictor - AI Impact Hackathon

This repository contains the solution developed for **Ford's Challenge** at the **AI Impact Hackathon**. The project focuses on predicting future vehicle Diagnostic Trouble Codes (DTCs) by analyzing historical fault sequences and vehicle telemetry.

##  Overview

Predicting vehicle faults before they become critical failures is a major challenge in automotive maintenance. To solve this, we built a hybrid predictive model that learns from sequential fault data to forecast the most likely trouble codes a vehicle will experience in the near future. 

##  Model Architecture

Our solution combines deep learning with statistical probability to maximize accuracy:
* **LSTM Neural Network:** A recurrent neural network that processes historical sequences of primary DTCs. It also incorporates engineered features like the time elapsed between faults (`delta_t`), odometer changes (`delta_odo`), and the global frequency of specific faults (`log_freq`).
* **Markov Chain Blending:** A Markov model calculates the historical transition probabilities from one fault to the next. These probabilities are blended directly with the LSTM's predictions (at a 40% weight) to ground the neural network's outputs in historical statistical reality.

##  Key Features & Data Pipeline

* **Smart Preprocessing:** The data pipeline handles irregular time intervals. If a vehicle experiences a gap of more than 30 days between faults, the system automatically injects a `[NO_FAULT]` token to accurately represent periods of healthy operation.
* **Sequence Windowing:** The model looks at a historical sequence length of 10 steps to predict faults that will occur within a future target window of 5 steps.
* **Imbalance Handling:** Because some faults are incredibly rare, the model trains using a Custom Focal Binary Cross-Entropy (BCE) Loss function (Gamma = 1.0) to prevent frequent faults from dominating the learning process.

##  Performance & Results

The model was evaluated using **Recall@5**, meaning the system is considered successful if the actual future fault appears in the model's top 5 predictions.

* **Validation R@5:** Peaked at ~0.8711 during training.
* **Final Test Score:** The fully blended LSTM + Markov model achieved a highly accurate **Recall@5 of 0.9287** on a strictly held-out test set of 20 completely unseen vehicles.

##  Tech Stack

* **Language:** Python 3
* **Core Libraries:** PyTorch (with CUDA support for GPU acceleration), Pandas, NumPy, Scikit-learn

##  How to Run

1. Clone the repository and install the required dependencies:
   ```bash
   pip install pandas numpy torch scikit-learn openpyxl
