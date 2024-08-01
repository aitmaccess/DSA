import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data[:, :2]
y = iris.target 

svm = SVC(kernel='linear')
svm.fit(X_iris, y)

DecisionBoundaryDisplay.from_estimator(svm, X_iris, response_method="predict")
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y, s=20, edgecolors="k")
plt.show()
