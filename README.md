# arXiv Paper Classification

Classification of scientific papers from arXiv into 8 topic categories using transformer models.

The dataset contains 12,800 papers from the Kaggle arXiv dataset. Each paper is represented by its title and abstract.

## Models

Two pretrained transformer models were fine-tuned and compared:

* DistilBERT
* SciBERT

| Model      | Validation Accuracy | Macro F1 |
| ---------- | ------------------: | -------: |
| DistilBERT |              85.47% |   85.47% |
| SciBERT    |              88.67% |   88.70% |

SciBERT was used as the final model.

## Categories

* Computer Science
* Economics
* Electrical Engineering and Systems Science
* Mathematics
* Physics
* Quantitative Biology
* Quantitative Finance
* Statistics

## App

A Streamlit application allows users to enter a paper title and/or abstract and returns the predicted topic probabilities.

## Files

* `training.ipynb` — data preparation, model training and evaluation
* `app.py` — Streamlit application
* `test_examples.csv` — example papers for testing the app
