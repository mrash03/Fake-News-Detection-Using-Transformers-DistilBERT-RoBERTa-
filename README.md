# Fake News Detection Using Transformers (DistilBERT & RoBERTa)

## Objective
Detect fake news using transformer-based NLP models on the ISOT and LIAR datasets.

## Datasets
- **ISOT:** Large binary-labeled dataset for fake news detection (Real/Fake).  
- **LIAR:** Smaller, challenging dataset with short political statements labeled True or False.  

## Method

### Notebook 1: DistilBERT Transfer Learning
- Trained DistilBERT on **ISOT dataset**: achieved near-perfect accuracy of **0.9998** on the test set.  
- When directly evaluated on **LIAR dataset**, initial accuracy dropped to **0.4372**.  
  - This drop is mainly due to **overfitting on ISOT** and domain differences between ISOT and LIAR.  
- Fine-tuned DistilBERT on LIAR to mitigate overfitting and improve generalization:
  - **Fine-tuned test accuracy:** 0.6433  
  - **Macro F1-score:** 0.63  

### Notebook 2: RoBERTa from Scratch on LIAR
- Trained a RoBERTa model from scratch on LIAR dataset.  
- Achieved final test accuracy of **0.6661** and F1-score **0.6666**.
- ### Confusion Matrix 

| True \ Pred | 0   | 1   |
|------------|-----|-----|
| 0          | 604 | 214 |
| 1          | 209 | 240 |



## Results Summary

| Model                    | Dataset | Accuracy | Macro F1-score |
|---------------------------|---------|----------|----------------|
| DistilBERT (ISOT)        | ISOT    | 0.9998   | 1.00           |
| DistilBERT (fine-tuned)  | LIAR    | 0.6433   | 0.63           |
| RoBERTa from scratch      | LIAR    | 0.6661   | 0.67       |

---

## How to Run
- Open the notebooks `DistilBERT_ISOT_LIAR.ipynb` and `RoBERTa_LIAR.ipynb` in Kaggle or Jupyter.  
- Run all the cells to produce the results. 
