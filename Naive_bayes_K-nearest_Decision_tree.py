from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer,Binarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load datasets
iris = load_iris()
wine = load_wine()

datasets = {
    "Iris": iris,
    "Wine": wine
}

classifiers = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}
# Function to scale data
def scale_data(X, y=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if y is not None:
        return X_scaled, y
    else:
        return X_scaled
    



# Function to evaluate classifiers
def evaluate_classifiers(X_train, X_test, y_train, y_test, cv, dataset_name, silent=False):
    accuracies = []
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        if not silent:
            print(f"{clf_name} Accuracy ({dataset_name}): {accuracy:.4f}")
    if cv:
        for clf_name, clf in classifiers.items():
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv)
            mean_accuracy = cv_scores.mean()
            print(f"{clf_name} Mean Accuracy ({dataset_name}, Cross-validation): {mean_accuracy:.4f}")
    return accuracies


# Function to perform classification under different training set selection methods
def compare_classifiers(dataset_name, train_method, cv=None):
    dataset = datasets[dataset_name]
    X, y = dataset.data, dataset.target
    
    if train_method == "holdout":
        for situation in ["75%-25%", "2/3 (66.6%)-1/3 (33.3%)"]:
            print(f"\nSituation: Training set = {situation}")
            if cv:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, test_size=1/3, random_state=42)
            X_train_scaled, X_test_scaled = scale_data(X_train), scale_data(X_test)
            evaluate_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, cv, dataset_name)
        
    elif train_method == "random_subsampling":
        for situation in ["75%-25%", "2/3 (66.6%)-1/3 (33.3%)"]:
            print(f"\nSituation: Training set = {situation}")
            accuracies = []
            for _ in range(5):  # Perform 5 iterations of random subsampling
                if cv:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=None)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, test_size=1/3, random_state=None)
                X_train_scaled, X_test_scaled = scale_data(X_train), scale_data(X_test)
                accuracies.extend(evaluate_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, cv=None, dataset_name=dataset_name, silent=True))
            if accuracies:  # Check if accuracies list is not empty
                print(f"Average Accuracy ({dataset_name}, Random Subsampling): {sum(accuracies) / len(accuracies):.4f}")
        
    elif train_method == "cross_validation":
        for situation in ["75%-25%", "2/3 (66.6%)-1/3 (33.3%)"]:
            print(f"\nSituation: Training set = {situation}")
            X_scaled, y_scaled = scale_data(X, y)  # Scale entire dataset for cross-validation
            for clf_name, clf in classifiers.items():
                cv_scores = cross_val_score(clf, X_scaled, y_scaled, cv=5)  # Using 5-fold cross-validation
                mean_accuracy = cv_scores.mean()
                print(f"{clf_name} Mean Accuracy ({dataset_name}, Cross-validation): {mean_accuracy:.4f}")
        print("\n")



for dataset_name in datasets.keys():
    print(f"Dataset: {dataset_name}\n")
    
    # Holdout method
    print("Holdout Method:")
    compare_classifiers(dataset_name, "holdout")
    
    # Random subsampling method
    print("\nRandom Subsampling Method:")
    compare_classifiers(dataset_name, "random_subsampling")
    
    # Cross-validation method
    print("Cross-validation Method:")
    compare_classifiers(dataset_name, "cross_validation")
