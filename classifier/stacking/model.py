from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


fp = r'Preprocessing\ouput\preprocessed_data_fixed.csv'


# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train and evaluate
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
print(f"Stacking Model Accuracy: {accuracy_score(y_test, y_pred)}")

