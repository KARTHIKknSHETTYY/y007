import joblib
from sklearn.datasets import load_iris      
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression().fit(X, y)
joblib.dump(model, 'iris_model.joblib')
print("Model trained and saved successfully as 'iris_model.joblib'")