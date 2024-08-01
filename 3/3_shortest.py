from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('/content/drive/MyDrive/Datasets_ml/PlayTennis.csv')
x = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data['Play Tennis']

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(x).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

new_sample = pd.DataFrame([['Overcast', 'Hot', 'High', 'Weak']], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
new_sample_encoded = encoder.transform(new_sample).toarray()
prediction = clf.predict(new_sample_encoded)

print("Prediction for the new sample:", prediction[0])
