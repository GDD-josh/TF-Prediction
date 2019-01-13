import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def run_SVM(X_train, X_test, y_train, y_test, X_valid, y_valid):
    # Impute data - NaN replaced with mean
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_valid = imp.transform(X_valid)
    X_test = imp.transform(X_test)

    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # Make predictions
    SVM_fscore = f1_score(y_test, y_pred, average="micro")
    print("\nLinear SVM F-score: {0}".format(SVM_fscore))


# Naive Bayes Classifier
def run_Naive_Bayes(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # Impute data - NaN replaced with mean
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_valid = imp.transform(X_valid)
    X_test = imp.transform(X_test)

    alpha = 1  # TODO: change this
    model = GaussianNB()  # Fit classifier to the data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)  # Make Predictions

    # Result
    nb_fscore = f1_score(y_valid, y_pred, average="micro")
    print("\nNaive Bayes: alpha = {1},    F-score: {0}".format(nb_fscore, alpha))


# Decision Tree Classifier
def run_Decision_Tree(X_train, y_train, X_valid, y_valid):
    # Impute data - NaN replaced with mean
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_valid = imp.transform(X_valid)

    model = DecisionTreeClassifier(random_state=1)  # Fit classifier to the data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)  # Make predictions
    dt_fscore = f1_score(y_valid, y_pred, average="micro")
    print("\nDecision Tree F-score: {0}".format(dt_fscore))


def select_svc_params(X, y):
    print("Launching SVM parameter calculation")
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    print("Best Score: {}".format(grid_search.best_score_))
    print("Best Params: {}".format(grid_search.best_estimator_))
    print("Best params: {}".format(grid_search.best_params_))


def select_decision_tree_params(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = DecisionTreeClassifier(random_state=1)  # Fit classifier to the data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # Make predictions

    # Result
    dt_fscore = f1_score(y_test, y_pred, average="micro")
    print("Baseline:\nDecision Tree F-score: {0}".format(dt_fscore))

    min_samples_splits = np.linspace(0.1, 0.7, 10, endpoint=True)

    parameter_grid = [{'criterion': ['entropy', 'gini'],
                       'max_depth': [6, 8, 10, 12, 15],
                       'min_samples_split': min_samples_splits}]

    grid_search = GridSearchCV(model, param_grid=parameter_grid, scoring='f1_micro', cv=5, n_jobs=-1)
    grid_search.fit(X_valid, y_valid)

    print("Best Score: {}".format(grid_search.best_score_))
    print("Best Params: {}".format(grid_search.best_estimator_))
    print("Best params: {}".format(grid_search.best_params_))


def flatten(dataset):
    newList = []
    for x in dataset:
        newList.append(np.concatenate(x).ravel())
    return newList


def main():
    # load data
    print("Loading data files")
    X_valid = np.load("data/validx_final.npy")
    X_train = np.load("data/trainx_final.npy")
    X_test = np.load("data/testx_final.npy")

    y_valid = np.load("data/validy_final.npy")
    y_train = np.load("data/trainy_final.npy")
    y_test = np.load("data/testy_final.npy")

    X_train = flatten(X_train)
    X_test = flatten(X_test)
    X_valid = flatten(X_valid)

    # Random Classifier
    rand_dummy = DummyClassifier(strategy='uniform', random_state=1)  # Fit classifier to the data
    rand_dummy.fit(X_train, y_train)
    y_pred = rand_dummy.predict(X_test)  # Make Predictions
    randClassifier_fscore = f1_score(y_test, y_pred, average="micro")
    print("\nRandom Classifier F-score: {0}".format(randClassifier_fscore))

    # Majority-Class Classifier
    major_dummy = DummyClassifier(strategy='most_frequent')  # Fit classifier to the data
    major_dummy.fit(X_train, y_train)
    y_pred = major_dummy.predict(X_test)  # Make Predictions
    majorityClass_fscore = f1_score(y_test, y_pred, average="micro")
    print("Majority Class Classifier F-score: {0}".format(majorityClass_fscore))

    # Gaussian Naive Bayes
    run_Naive_Bayes(X_train, y_train, X_valid, y_valid, X_test, y_test)

    # Decision Tree
    run_Decision_Tree(X_train, y_train, X_valid, y_valid)

    # SVM
    run_SVM(X_train, y_train, X_valid, y_valid, X_test, y_test)


main()
