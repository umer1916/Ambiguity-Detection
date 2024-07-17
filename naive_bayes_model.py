import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
import scipy
import joblib
dataset_path = r""

# Load the Excel file
file_path = os.path.join(dataset_path, "")
df = pd.read_excel(file_path)


# Columns to be used as features (X)
feature_columns = [
    'Sentence', 'Noun Phrase', 'Pronoun', 'Number Agreement', 'Definiteness', 'Non-prepositional', 
    'Syntactic Constraint', 'Syntactic Parallelism', 'Coordination Pattern', 'Non-associated', 
    'Indicating Verb', 'Semantic Constraint', 'Semantic Parallelism', 'Domain-specific Term', 
    'Centering', 'Section Heading', 'Sentence Recency', 'Proximal', 'Local Collocation Frequency', 
    'BNC Collocation Frequency'
]

# Use the specified columns as features (X)
X = df[feature_columns]
# Use the 'Ambiguous' column as the target variable (y)
y = df['Ambigious ']

# Combine textual features into a single column
df['Combined Text'] = df['Sentence'] + " " + df['Noun Phrase'] + " " + df['Pronoun']

# Vectorize the combined textual features
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(df['Combined Text'])

# Encode heuristics features as numbers and store them in a combined list
heuristics_features = [
    'Number Agreement', 'Definiteness', 'Non-prepositional', 'Syntactic Constraint', 'Syntactic Parallelism', 
'Coordination Pattern', 'Non-associated', 'Indicating Verb', 'Semantic Constraint', 'Semantic Parallelism', 
    'Domain-specific Term', 'Centering', 'Section Heading', 'Sentence Recency', 'Proximal', 
    'Local Collocation Frequency', 'BNC Collocation Frequency'
]

label_encoders = {}
combined_heuristics = []
for index, row in df.iterrows():
    combined_value = []
    for feature in heuristics_features:
        if feature not in label_encoders:
            label_encoders[feature] = LabelEncoder()
            label_encoders[feature].fit(df[feature])
        combined_value.append(label_encoders[feature].transform([row[feature]])[0])
    combined_heuristics.append(combined_value)

# Combine textual features and heuristics features
X_combined = scipy.sparse.hstack([X_text, scipy.sparse.csr_matrix(combined_heuristics)])

# Encode the target variable (y)
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)
# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)

# Save the trained model
model_file = "naive_bayes_model_combined.joblib"
joblib.dump(model, model_file)

# Output the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1-score: {f1}")
print(f"Classification Report:\n{report}")
print(f"Model saved to: {model_file}")
