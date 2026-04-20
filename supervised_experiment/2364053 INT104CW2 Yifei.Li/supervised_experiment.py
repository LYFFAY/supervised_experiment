import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score
from classifier_functions import train_knn_classifier, train_naive_bayes_classifier, train_decision_tree_classifier
from classifier_functions import train_ensemble_classifier
from evaluate import evaluate_classification
# Load Data
file_path = "C:/Users/w/PycharmProjects/PythonProject6/training_set.xlsx"
data = pd.read_excel(file_path)
feature_matrix = data.iloc[:,1:]
label = data.iloc[:,0]

# Scale Data
scaler_1 = StandardScaler()
scaler_2 = MinMaxScaler()
scaler_3 = Normalizer()
sd_feature_1 = scaler_1.fit_transform(feature_matrix)
sd_feature_2 = scaler_2.fit_transform(feature_matrix)
sd_feature_3 = scaler_3.fit_transform(feature_matrix)

train_sd1, test_sd1, y_sd1, y_test_sd1 = train_test_split(sd_feature_1, label, test_size=0.2, random_state=42)
train_sd2, test_sd2, y_sd2, y_test_sd2 = train_test_split(sd_feature_2, label, test_size=0.2, random_state=42)
train_sd3, test_sd3, y_sd3, y_test_sd3 = train_test_split(sd_feature_3, label, test_size=0.2, random_state=42)

# kNN experiments
knn3 = train_knn_classifier(train_sd1, y_sd1, n_neighbors=3)
knn5 = train_knn_classifier(train_sd1, y_sd1, n_neighbors=5)
knn3_pred = knn3.predict(test_sd1)
knn5_pred = knn5.predict(test_sd1)
knn3_accuracy = accuracy_score(y_test_sd1, knn3_pred)
knn5_accuracy = accuracy_score(y_test_sd1, knn5_pred)

# Decision tree experiments
dt_gini = train_decision_tree_classifier(train_sd2, y_sd2, dt_criterion='gini')
dt_entropy = train_decision_tree_classifier(train_sd2, y_sd2, dt_criterion='entropy')
dtg_pred = dt_gini.predict(test_sd2)
dte_pred = dt_entropy.predict(test_sd2)
dtg_accuracy = accuracy_score(y_test_sd2, dtg_pred)
dte_accuracy = accuracy_score(y_test_sd2, dte_pred)

# Naive Bayes experiments
nb_random = train_naive_bayes_classifier(train_sd3, y_sd3, priors=[0.25, 0.25, 0.25, 0.25])
nb_prior = train_naive_bayes_classifier(train_sd3, y_sd3, priors=[0.43, 0.36, 0.06, 0.15])
nbr_pred = nb_random.predict(test_sd3)
nbp_pred = nb_prior.predict(test_sd3)
nbr_accuracy = accuracy_score(y_test_sd3, nbr_pred)
nbp_accuracy = accuracy_score(y_test_sd3, nbp_pred)

# Show results of single classifier
print(f'for kNN, k=3 accuracy: {knn3_accuracy:.2f}, k=5 accuracy: {knn5_accuracy:.2f}')
print(f'for decision tree, gini accuracy: {dtg_accuracy:.2f}, entropy accuracy: {dte_accuracy:.2f}')
print(f'for naive Bayes, random accuracy: {nbr_accuracy:.2f}, prior accuracy: {nbp_accuracy:.2f}')

# Ensemble Classifier Experiments
enC_hard = train_ensemble_classifier(train_sd1, y_sd1, knn5, nb_prior, dt_entropy, ec_voting='hard')
enC_soft = train_ensemble_classifier(train_sd1, y_sd1, knn5, nb_prior, dt_entropy, ec_voting='soft')

ech_pred = enC_hard.predict(test_sd1)
ech_accuracy = accuracy_score(y_test_sd1, ech_pred)
ecs_pred = enC_soft.predict(test_sd1)
ecs_accuracy = accuracy_score(y_test_sd1, ecs_pred)

print(f'for ensemble learning, hard accuracy: {ech_accuracy:.2f}, soft accuracy: {ecs_accuracy:.2f}')


# Load Test Data
test_file_path = "C:/Users/w/PycharmProjects/PythonProject6/training_set.xlsx"
test_data = pd.read_excel(test_file_path)
test_data_feature = test_data.iloc[:, 1:]

#Transform Data to obtain Features
sd_feature_test = scaler_1.transform(test_data_feature)
#Train Classifier with Training Data
#dt_entropy_new = train_decision_tree_classifier(sd_feature_3, label, dt_criterion='entropy')
#Evaluate the classifier
y_pred = enC_hard.predict(sd_feature_test)

evaluate_classification(y_pred, "check_Monday.bin")