from extract_features import load_training_data, load_tests
from sklearn.svm import LinearSVC


data, labels = load_training_data()

model = LinearSVC()
model.fit(data, labels)



a = model.predict([data[0]])
print(a)