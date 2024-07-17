import os
import json
import logging
from flask import Flask, redirect, render_template, request, jsonify, send_from_directory, url_for
import joblib
import scipy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from werkzeug.utils import secure_filename
import pandas as pd
from new_code_heuristics import allowed_file, preprocess_text
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
MODEL_SAVE_PATH = 'saved_models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/train_model')
def train_model():
    return render_template('Train ML Model.html')

@app.route('/train_model_api', methods=['POST'])
def train_model_api():
    try:
        data = request.get_json()
        file_data = data.get('fileData')
        model_type = data.get('modelType')
        
        # Save the uploaded file data
        file_path = 'uploaded_file.xlsx'
        with open(file_path, 'wb') as f:
            f.write(file_data.encode('latin1'))

        # Load the Excel file
        df = pd.read_excel(file_path)

        # Columns to be used as features (X)
        feature_columns = [
            'Sentence', 'Noun Phrase', 'Pronoun', 'Number Agreement', 'Definiteness', 'Non-prepositional', 
            'Syntactic Constraint', 'Syntactic Parallelism', 'Coordination Pattern', 'Non-associated', 
            'Indicating Verb', 'Semantic Constraint', 'Semantic Parallelism', 'Domain-specific Term', 
            'Centering', 'Section Heading', 'Sentence Recency', 'Proximal', 'Local Collocation Frequency', 
            'BNC Collocation Frequency'
        ]

        X = df[feature_columns]
        y = df['Ambigious ']

        df['Combined Text'] = df['Sentence'] + " " + df['Noun Phrase'] + " " + df['Pronoun']

        vectorizer = CountVectorizer()
        X_text = vectorizer.fit_transform(df['Combined Text'])

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

                transformed_value = label_encoders[feature].transform([row[feature]])[0]
                combined_value.append(transformed_value)

            combined_heuristics.append(combined_value)

        combined_heuristics = scipy.sparse.csr_matrix(combined_heuristics)
        X_combined = scipy.sparse.hstack([X_text, combined_heuristics])

        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

        if model_type == 'GaussianNB':
            model = GaussianNB()
        elif model_type == 'MultinomialNB':
            model = MultinomialNB()
        else:
            return jsonify({'error': 'Invalid model type. Please choose either GaussianNB or MultinomialNB.'})

        model.fit(X_train.toarray(), y_train)
        y_pred = model.predict(X_test.toarray())
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        classification_rep = classification_report(y_test, y_pred)

        model_save_path = os.path.join(MODEL_SAVE_PATH, f'{model_type}.joblib')
        joblib.dump(model, model_save_path)
        vectorizer_save_path = os.path.join(MODEL_SAVE_PATH, 'vectorizer.joblib')
        joblib.dump(vectorizer, vectorizer_save_path)
        label_encoders_save_path = os.path.join(MODEL_SAVE_PATH, 'label_encoders.joblib')
        joblib.dump(label_encoders, label_encoders_save_path)

        return jsonify({
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1,
            'classification_report': classification_rep
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/file_info')
def file_info():
    return render_template('preprocessfileinfo.html')

@app.route('/save_model_form', methods=['POST'])
def save_model_form():
    model_name = request.form.get('modelName')
    model_data = request.form.get('modelData')
    return render_template('Save Model.html', modelName=model_name, modelData=model_data)

@app.route('/save_model', methods=['POST'])
def save_model():
    model_name = request.form.get('modelName')
    # Additional logic to save the model
    return "Model saved successfully!"

@app.route('/load_model')
def load_model():
    return render_template('Load Model.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.json'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/file_details')
def file_details():
    return render_template('file-details.html')

@app.route('/view_file/<filename>', methods=['GET'])
def view_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()
        return render_template('view_file.html', filename=filename, content=content)
    else:
        return jsonify({'error': 'File not found'}), 404

def train_naive_bayes_model():
    # Placeholder for actual model training logic
    return {"modelName": "Naive Bayes Algorithm", "accuracy": 0.95}

@app.route('/detect_ambiguity', methods=['GET'])
def detect_ambiguity_form():
    return render_template('Detect Ambiguity.html')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/detect_ambiguity', methods=['POST'])
def detect_ambiguity():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            df = pd.read_excel(file_path)

            model_path = os.path.join(MODEL_SAVE_PATH, 'MultinomialNB.joblib')
            vectorizer_path = os.path.join(MODEL_SAVE_PATH, 'vectorizer.joblib')
            label_encoders_path = os.path.join(MODEL_SAVE_PATH, 'label_encoders.joblib')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            if not os.path.exists(label_encoders_path):
                raise FileNotFoundError(f"Label encoders file not found: {label_encoders_path}")

            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            label_encoders = joblib.load(label_encoders_path)

            df['Combined Text'] = df['Sentence'] + " " + df['Noun Phrase'] + " " + df['Pronoun']
            X_text = vectorizer.transform(df['Combined Text'])

            heuristics_features = [
                'Number Agreement', 'Definiteness', 'Non-prepositional', 'Syntactic Constraint', 'Syntactic Parallelism', 
                'Coordination Pattern', 'Non-associated', 'Indicating Verb', 'Semantic Constraint', 'Semantic Parallelism', 
                'Domain-specific Term', 'Centering', 'Section Heading', 'Sentence Recency', 'Proximal', 
                'Local Collocation Frequency', 'BNC Collocation Frequency'
            ]

            combined_heuristics = []
            for index, row in df.iterrows():
                combined_value = []
                for feature in heuristics_features:
                    transformed_value = label_encoders[feature].transform([row[feature]])[0]
                    combined_value.append(transformed_value)

                combined_heuristics.append(combined_value)

            combined_heuristics = scipy.sparse.csr_matrix(combined_heuristics)
            X_combined = scipy.sparse.hstack([X_text, combined_heuristics])

            predictions = model.predict(X_combined.toarray())

            results = []
            for sentence, pred in zip(df['Sentence'], predictions):
                results.append({'Sentence': sentence, 'Predicted Ambiguity': pred})

            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Allowed file types are xlsx.'})
@app.route('/download')
def download_file():
    try:
        return send_from_directory(directory='uploads', path='Ambiguity_Detection_Output_for_processed_results.xlsx', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
