# 📰 AG News Classification with Transformer

A **Transformer-based** text classifier trained from scratch on the AG News dataset to categorize news articles into 4 topics.

---

## 🧠 Model Architecture

- **Tokenizer:** Custom BPE (Byte Pair Encoding), vocab size: 16,000
- **Embedding:** Token embedding + Sinusoidal Positional Encoding
- **Encoder:** Transformer Encoder (3 layers, 8 heads)
- **Embedding dim:** 256
- **Feedforward dim:** 512
- **Dropout:** 0.1
- **Output:** Linear layer → 4 classes

---

## 📦 Dataset

- **Source:** [Kaggle — AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Files:** `train.csv`, `test.csv`

| Class Index | Category |
|-------------|----------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

| Split      | Samples |
|------------|---------|
| Train      | 112,400 |
| Validation | 7,600   |
| Test       | 7,600   |

---

## ⚙️ Training Details

| Parameter       | Value    |
|-----------------|----------|
| Optimizer       | Adam     |
| Learning rate   | `0.0001` |
| Batch size      | `32`     |
| Epochs          | `5`      |
| Loss function   | CrossEntropyLoss |
| Device          | CUDA / CPU (auto) |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/edhaevv/ag-news-classification.git
cd ag-news-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get the dataset from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) and place `train.csv` and `test.csv` in `/content/` (or update the paths in the notebook).

### 4. Run the notebook
Open `AGNewsClassification.ipynb` in Jupyter or Google Colab and run all cells.

---

## 🔍 Inference

```python
sample_text = "The new smartphone features a powerful processor and advanced camera system."
result = predict(model, tokenizer, sample_text, device)
print(f"Predicted Class: {result}")  # → Sci/Tech
```

---

## 📁 Project Structure

```
├── AGNewsClassification.ipynb   # Main notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files
└── README.md                    # This file
```

---

## 🛠️ Requirements

See `requirements.txt`. Main dependencies:

- `torch`
- `tokenizers`
- `pandas`
- `tqdm`
- `sympy`
