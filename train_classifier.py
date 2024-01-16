import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Compute overall accuracy
score = accuracy_score(y_predict, y_test)
print('Overall accuracy: {}% of samples were classified correctly!'.format(score * 100))

# Compute accuracy for each class
unique_classes = np.unique(labels)
for class_label in unique_classes:
    indices = np.where(y_test == class_label)
    class_accuracy = accuracy_score(y_predict[indices], y_test[indices])
    print('Class {} accuracy: {}%'.format(class_label, class_accuracy * 100))

# Save the model directly
with open('model.p', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
