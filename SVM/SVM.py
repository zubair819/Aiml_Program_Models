#Step 1: import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Step 2: load the iris dataset
iris = datasets.load_iris()

#Step 3: split te datased into training and testing sets
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Step 4: Instantiate and fit the SVM model on the training data
model = SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train)

#step 5: Predict the lable and train the model (accuracy) for the test set
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#Step 6: Pot the decision boundaries for the 2D projection of the dataset
def plot_decision_boundaries(x, y, model):
    h = .02 #step size in the mesh
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contourf(xx, yy, z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundaried')
    plt.show()
    
plot_decision_boundaries(x, y, model)