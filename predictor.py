import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. GENERATE MOCK HISTORICAL DATA
# ==========================================
# Simulating 1000 past football matches
# Features: Home Possession (%), Home Shots on Target, Away Shots on Target
# Target (Result): 2 = Home Win, 1 = Draw, 0 = Away Win

print("Generating historical match data...")
np.random.seed(42)

data_size = 1000
home_possession = np.random.randint(30, 70, data_size)
home_shots = np.random.randint(0, 10, data_size)
away_shots = np.random.randint(0, 10, data_size)

# Simple logic to define the match result based on stats
results = []
for i in range(data_size):
    score = (home_possession[i] * 0.1) + (home_shots[i] * 2) - (away_shots[i] * 2)
    if score > 8:
        results.append(2) # Home Win
    elif score < 4:
        results.append(0) # Away Win
    else:
        results.append(1) # Draw

# Create a DataFrame
df = pd.DataFrame({
    'Home_Possession': home_possession,
    'Home_Shots_On_Target': home_shots,
    'Away_Shots_On_Target': away_shots,
    'Result': results
})

# ==========================================
# 2. TRAIN THE MACHINE LEARNING MODEL
# ==========================================
print("Training the Random Forest model...")

# Split Features (X) and Target (y)
X = df[['Home_Possession', 'Home_Shots_On_Target', 'Away_Shots_On_Target']]
y = df['Result']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model's accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%\n")

# ==========================================
# 3. MAKE A PREDICTION ON A NEW MATCH
# ==========================================
# Imagine a new match: Home team has 60% possession, 7 shots on target. Away team has 2 shots.
new_match = pd.DataFrame({
    'Home_Possession': [60],
    'Home_Shots_On_Target': [7],
    'Away_Shots_On_Target': [2]
})

predicted_result = model.predict(new_match)

print("--- NEW MATCH PREDICTION ---")
print("Home Possession: 60% | Home Shots: 7 | Away Shots: 2")

if predicted_result[0] == 2:
    print("🤖 AI Prediction: HOME TEAM WINS")
elif predicted_result[0] == 1:
    print("🤖 AI Prediction: DRAW")
else:
    print("🤖 AI Prediction: AWAY TEAM WINS")
