import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

file_path = 'Heuristic_Evaluation_Output.xlsx'
data = pd.read_excel(file_path)

label_encoders = {}

for column in data.columns[1:]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

X = data.drop(['Centering', 'Sentence'], axis=1)  
y = data['Centering']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

data['Predicted Ambiguity'] = model.predict(X)

data['Predicted Ambiguity'] = data['Predicted Ambiguity'].map({1: 'Y', 0: 'N'})

for column in data.columns[1:-1]:  
    le = label_encoders[column]
    data[column] = le.inverse_transform(data[column])

output_path = 'Ambiguity_Detection_Output_for_processed_results.xlsx'
data.to_excel(output_path, index=False)

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS