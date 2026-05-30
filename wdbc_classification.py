from sklearn import datasets
from sklearn import svm, tree, neighbors, ensemble, metrics, model_selection
import matplotlib.pyplot as plt
import numpy as np

wdbc = datasets.load_breast_cancer()

X = wdbc.data
y = wdbc.target

# getting the data schema
print("---DESCRIPTION---")
print("Dataset shape: ", X.shape)
print("Target shape: ", y.shape)
print("Features: ", wdbc.feature_names)
print("Target names: ", wdbc.target_names)
print("First 5 samples: ", X[:5])
print("First 5 targets: ", y[:5])
print("First 5 feature names: ", wdbc.feature_names[:5])
print("Target names: ", wdbc.target_names)
print("-----------------")

# initializing the models
models = {
    "SVM": svm.SVC(),
    "Decision Tree": tree.DecisionTreeClassifier(random_state = 42),
    "K-Nearest Neighbors": neighbors.KNeighborsClassifier(),
    "Random Forest": ensemble.RandomForestClassifier(random_state = 42)
}

# initializing the dictionaries for the best model purposes
model_accuracies = {"SVM": 0, "Decision Tree": 0, "K-Nearest Neighbors": 0, "Random Forest": 0}
model_y_test = {"SVM": [], "Decision Tree": [], "K-Nearest Neighbors": [], "Random Forest": []}
model_y_pred = {"SVM": [], "Decision Tree": [], "K-Nearest Neighbors": [], "Random Forest": []}

# training the models and retrieving the metrics
for name, model in models.items():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    model_y_test[name] = y_test
    model_y_pred[name] = y_pred
    
    print("---METRICS---")
    print(f"{name}: Accuracy = {accuracy:.4f}")
    print(f"Recall = {metrics.recall_score(y_test, y_pred):.4f}")
    print(f"Precision = {metrics.precision_score(y_test, y_pred):.4f}")
    print(f"F1-Score = {metrics.f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix = {metrics.confusion_matrix(y_test, y_pred)}")
    print("-------------")

# choosing the model with the highest accuracy
print("\n---BEST MODEL---")
best_model_name = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model_name} with Accuracy = {model_accuracies[best_model_name]:.4f}")
print("------------------")

# plotting the confusion matrix for the best model
y_test = model_y_test[best_model_name]
y_pred = model_y_pred[best_model_name]
cm = metrics.confusion_matrix(y_test, y_pred)

plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title(f"{best_model_name} Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(wdbc.target_names))
plt.xticks(tick_marks, wdbc.target_names, rotation = 45)
plt.yticks(tick_marks, wdbc.target_names)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                    ha = "center",
                    va = "center",
                    color = "white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# creating a scatter plot
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu)
plt.title("Scatter Plot of Mean Radius vs Mean Texture")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.show()