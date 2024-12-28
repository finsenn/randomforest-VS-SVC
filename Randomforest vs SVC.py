import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import glob

# Base path for datasets
dataset_dir = 'C:/Users/Wesley Benedict/Desktop/RESERCH/web_bot_detection_dataset/phase1/data'

# Paths for web logs and mouse movements
web_logs_dir = os.path.join(dataset_dir, 'web_logs')
mouse_movements_dir = os.path.join(dataset_dir, 'mouse_movements')

# Function to read data from bots and humans subdirectories
def read_log_data(base_dir):
    bots_dir = os.path.join(base_dir, 'bots')
    humans_dir = os.path.join(base_dir, 'humans')
    bot_data, human_data = [], []

    # Read bot logs
    if os.path.exists(bots_dir):
        bot_files = glob.glob(os.path.join(bots_dir, '*.log'))
        for file in bot_files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    bot_data.append(line.strip())

    # Read human logs
    if os.path.exists(humans_dir):
        human_files = glob.glob(os.path.join(humans_dir, '*.log'))
        for file in human_files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    human_data.append(line.strip())
    
    return bot_data, human_data

# Step 1: Read web log data
print("Reading web logs...")
web_bot_data, web_human_data = read_log_data(web_logs_dir)
print(f"Web logs - Bots: {len(web_bot_data)}, Humans: {len(web_human_data)}")

# Step 2: Read mouse movement data
print("Reading mouse movements...")
mouse_bot_data, mouse_human_data = read_log_data(mouse_movements_dir)
print(f"Mouse movements - Bots: {len(mouse_bot_data)}, Humans: {len(mouse_human_data)}")

# Step 3: Combine all data
data = web_bot_data + web_human_data + mouse_bot_data + mouse_human_data
labels = [1] * (len(web_bot_data) + len(mouse_bot_data)) + [0] * (len(web_human_data) + len(mouse_human_data))

# Step 4: Prepare DataFrame
df = pd.DataFrame(data, columns=["log_data"])
df['label'] = labels

# Step 5: Feature extraction using TfidfVectorizer and custom features
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)  # Limit features to reduce memory usage
X_tfidf = vectorizer.fit_transform(df['log_data'])

# Additional custom features
df['log_length'] = df['log_data'].apply(len)
df['log_contains_bot'] = df['log_data'].apply(lambda x: 1 if 'bot' in x.lower() else 0)

# Combine features into a sparse matrix to save memory
X_custom = df[['log_length', 'log_contains_bot']].values
X_combined = hstack((X_tfidf, X_custom))

# Features and labels
X = X_combined
y = df['label']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize classifiers with simplified parameters to save resources
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Reduced depth and trees
svc_model = SVC(kernel='linear', C=1, random_state=42)  # Linear kernel for efficiency

# Train models
print("Training RandomForestClassifier...")
rf_model.fit(X_train, y_train)

print("Training SVC...")
svc_model.fit(X_train, y_train)

# Step 8: Make predictions
rf_predictions = rf_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)

# Step 9: Evaluate models
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Results")
    print(f"{'Metric':<15}{'Score'}")
    print(f"{'Accuracy':<15}{accuracy:.2%}")
    print(f"{'Precision':<15}{precision:.2%}")
    print(f"{'Recall':<15}{recall:.2%}")
    print(f"{'F1 Score':<15}{f1:.2%}\n")

evaluate_model("Random Forest", y_test, rf_predictions)
evaluate_model("Support Vector Classifier", y_test, svc_predictions)
