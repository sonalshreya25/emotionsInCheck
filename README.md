# EmotionsInCheck
# Emotion Classification with DistilBERT
Fine-Tuning a Small Language Model for sentiment analysis by classifying emotions

This project fine-tunes the lightweight transformer model DistilBERT on the dair-ai/emotion dataset.
Our goal is to build an efficient and accurate model capable of classifying text into six emotion categories:

* Sadness

* Joy

* Love

* Anger

* Fear

* Surprise

We started with attempts to fine-tune Pythia-1B using LoRA (Parameter-Efficient Fine-Tuning) but due to hardware limitations and time constraints, we pivoted to full fine-tuning of DistilBERT, a smaller, faster model that still delivers excellent performance.

## Directory Structure# Emotion Classification

##  Directory Structure
```
EmotionClassification/
├── EmotionClassification.ipynb  # Main training and evaluation notebook  
├── README.md                    # Project documentation  
├── requirements.txt             # Pinned dependencies  
└── wandb/                       # W&B logs and runs
```

---

##  Setup Instructions

### Clone the repository
```bash
git clone https://github.com/your-username/emotion-classification.git
cd emotion-classification
```

### Install required packages
```bash
pip install -r requirements.txt
```

### Setup Weights and Biases (Optional but recommended)
```bash
pip install wandb
wandb login
```

### Run the Notebook
Launch **Jupyter** or **Colab**, open `EmotionClassification.ipynb`, and run the cells step-by-step.

---

## 🛠️ Key Components

- **Dataset**: `dair-ai/emotion`
- **Model**: `DistilBERT-base-uncased`
- **Fine-tuning Strategy**: Full fine-tuning with weighted loss to handle class imbalance
- **Evaluation Metrics**:
  - Accuracy
  - Weighted F1 Score
  - Precision and Recall
  - ROC-AUC (One-vs-Rest)
  - Confusion Matrix

---

## 📈 Results

| Metric                  | Value   |
|--------------------------|---------|
| Test Accuracy            | 92.6%   |
| Test Weighted F1 Score    | 92.6%   |
| Weighted ROC-AUC Score    | 95.44%  |

- The model shows strong generalization with minimal overfitting.
- It handles all emotion classes well despite imbalanced training data.

---

## 💡 Future Improvements

- Use advanced data augmentation techniques like back translation for minority classes.
- Explore PEFT methods like LoRA or QLoRA on larger models using better hardware.
- Implement early stopping and model checkpointing to improve training stability.




