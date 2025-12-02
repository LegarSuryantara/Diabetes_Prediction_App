import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
import pickle


col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('diabetes.csv', names=col_names)

X = df.drop('Outcome', axis=1)
y = df['Outcome']


scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, stratify=y, random_state=2)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
X_train_prediction = knn_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print(f'Akurasi Data Training dengan KNN: {training_data_accuracy}')

pickle.dump(knn_model, open('diabetes_model_knn.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

print("Model KNN berhasil disimpan sebagai 'diabetes_model_knn.sav'!")