import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated NFL game stats (replace with real data for production)
data = {
    'home_team_pts': [24, 30, 21, 28, 17, 14, 35, 10, 27, 32],
    'away_team_pts': [14, 20, 28, 21, 23, 10, 24, 19, 17, 30],
    'home_team_yards': [320, 400, 275, 370, 280, 210, 410, 180, 350, 360],
    'away_team_yards': [280, 350, 310, 290, 260, 205, 380, 220, 300, 355],
    'home_turnovers': [1, 0, 2, 0, 1, 2, 1, 3, 1, 0],
    'away_turnovers': [2, 1, 1, 3, 0, 1, 2, 2, 2, 1],
}
df = pd.DataFrame(data)
df['home_win'] = (df['home_team_pts'] > df['away_team_pts']).astype(int)

X = df.drop(['home_team_pts', 'away_team_pts', 'home_win'], axis=1)
y = df['home_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

# Chart
plt.figure(figsize=(8,5))
plt.bar(
    range(len(probs)), probs,
    color=['green' if actual == 1 else 'red' for actual in y_test]
)
plt.xticks(range(len(probs)), [f'Game {i+1}' for i in range(len(probs))], rotation=45)
plt.ylabel('Predicted Home Win Probability')
plt.title('NFL Home Team Win Probabilities (Model Output)')
plt.tight_layout()
plt.savefig('nfl_home_win_probabilities.png')
plt.show()