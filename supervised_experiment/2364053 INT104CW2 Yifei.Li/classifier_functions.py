import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


def train_naive_bayes_classifier(features, label, priors=None):
    # Create a Gaussian Naive Bayes classifier
    model = GaussianNB(priors = priors)
    # Train the model
    model.fit(features,label)
    # Calculate accuracy of the training dataset
    y_pred = model.predict(features)
    accuracy = accuracy_score(label,y_pred)
    print(f'Naive Bayes Classifier has achieves an accuracy of {accuracy:.2f} for the training dataset.')
    return model

def train_decision_tree_classifier(features, label, dt_criterion='gini'):
    # Create a Decision Tree classifier
    model = DecisionTreeClassifier(criterion=dt_criterion,random_state=104)
    # Train the model
    model.fit(features,label)
    # Calculate accuracy of the training dataset
    y_pred = model.predict(features)
    accuracy = accuracy_score(label,y_pred)
    print(f'Decision Tree has achieve an accuracy of {accuracy:.2f} for the training dataset.')
    return model
def train_knn_classifier(features, label, n_neighbors=3):
    # Create a KNN classifier
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Train the model
    model.fit(features,label)
    # Calculate accuracy of the training dataset
    y_pred = model.predict(features)
    accuracy = accuracy_score(label, y_pred)
    print(f'KNN has achieved an accuracy of {accuracy: .2f} for the training dataset.')
    return model

def train_ensemble_classifier(features, label, knn, nb, dt, ec_voting='hard'):
    # Create an ensemble classifier using majority voting
    ensemble_model = VotingClassifier(estimators=[
        ('knn',knn),
        ('naive_bayes', nb),
        ('decision_tree', dt )
    ], voting = ec_voting)
    # Train the ensemble model
    ensemble_model.fit(features,label)
    # Calculate accuracy of the training dataset
    y_pred = ensemble_model.predict(features)
    accuracy = accuracy_score(label,y_pred)
    print(f'Ensemble Model has achieved an accuracy of {accuracy: .2f} for the training dataset.')
    return ensemble_model



