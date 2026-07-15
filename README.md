# ⚽ Football Match Predictor (Machine Learning)

## 📌 About The Project
As the co-founder of a sports betting mobile app (*Tebrus*), I am deeply interested in the intersection of sports data and Artificial Intelligence. 

This project is a lightweight Machine Learning script written in Python that predicts the outcome of a football match (Home Win, Draw, or Away Win) based on in-game statistics. 

## ⚙️ How it works
1. **Data Generation:** The script generates a mock dataset of 1000 past football matches with features such as Ball Possession, Home Shots on Target, and Away Shots on Target.
2. **Model Training:** It uses the **Scikit-Learn** library to train a `RandomForestClassifier` on 80% of the data.
3. **Prediction:** The model is evaluated on the remaining 20% of the data and then predicts the outcome of a completely new, unseen match.

## 💻 Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-Learn

## 🚀 Run the code
If you want to run this script locally:
```bash
pip install pandas numpy scikit-learn
python predictor.py
