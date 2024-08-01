import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[:, :2]
y = iris.target 

svm = SVC(kernel='linear')
svm.fit(x, y)

DecisionBoundaryDisplay.from_estimator(svm, x, response_method="predict")
plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolors="k")
plt.show()
