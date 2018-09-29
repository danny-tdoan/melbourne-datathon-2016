from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

import custom_csv as customcsv
import feature_extractor as fe
import xgboost as xgb
from xgboost import XGBClassifier

import numpy as np

import feature_words as fw


def get_gini_score_from_auc_score(AUCScore):
    return 2 * (AUCScore - 0.5)


class proba_extratrees(ExtraTreesClassifier):
    def predict(self, X):
        return ExtraTreesClassifier.predict_proba(self, X)


class proba_xgb(XGBClassifier):
    def predict(self, X):
        return XGBClassifier.predict_proba(self, X)


(all_job_ids_from_testing_data,
 training_vector_title_representation,
 training_vector_abstract_representation,
 training_data_prediction_col_values,
 testing_vector_title_representation,
 testing_vector_abstract_representation) = generate_training_and_test_vectors()

print("Doing Grid Search for XGBoost...")
xgb_model = xgb.XGBClassifier()
print(xgb_model.get_params().keys())
clf = GridSearchCV(xgb_model,
                   {'max_depth': [3, 6],
                    'n_estimators': [350, 400, 450, 500, 1000],
                    'objective': ['binary:logistic'],
                    'learning_rate': [0.1, 0.3],
                    'nthread': [3, 4]},
                   verbose=1, scoring='roc_auc')
clf.fit(np.array(training_vector_title_representation), np.array(training_data_prediction_col_values))

print('Grid Search XGBoost Best Score ' + str(clf.best_score_))
print('Grid Search XGBoost Best Params ' + str(clf.best_params_))

print("Cross-validating on XGBoost Title Features")
xgb_model = proba_xgb(clf.best_params_)
xgb_predictions_training_data_title_features = cross_val_predict(xgb_model,
                                                                 training_vector_title_representation,
                                                                 training_data_prediction_col_values,
                                                                 cv=NUM_CROSS_VALIDATION_FOLDS)

xgb_roc_score_title_features_model = roc_auc_score(training_data_prediction_col_values,
                                                   xgb_predictions_training_data_title_features[:, 1])
print('Cross-validated Title Features XGBoost Model Gini Score '
      + str(get_gini_score_from_auc_score(xgb_roc_score_title_features_model)) + '\n')

# Cross-validation for Normal Predictors Title Representation
print("Cross-validating on Extra Trees Title Features")
predictions_training_data_title_features = cross_val_predict(proba_extratrees(),
                                                             training_vector_title_representation,
                                                             training_data_prediction_col_values,
                                                             cv=NUM_CROSS_VALIDATION_FOLDS)

roc_score_title_features_model = roc_auc_score(training_data_prediction_col_values,
                                               predictions_training_data_title_features[:, 1])
print('Cross-validated Title Features Extra Trees Model Gini Score '
      + str(get_gini_score_from_auc_score(roc_score_title_features_model)) + '\n')


# **********************************DOING ACTUAL PREDICTION FOR -1 ROWS*********************************************
# Make prediction based on the title of the job posting

clf_random_forest_title = RandomForestClassifier()

print("Fitting model to complete title training data")
clf_random_forest_title.fit(training_vector_title_representation, training_data_prediction_col_values)

print("Generating predictions using fitted model for title data")
predicted_values1 = clf_random_forest_title.predict_proba(testing_vector_title_representation)

# Make prediction based on the abstract of the job posting
clf_random_forest_abstract = RandomForestClassifier()
print("Fitting model to complete abstract training data")
clf_random_forest_abstract.fit(training_vector_abstract_representation, training_data_prediction_col_values)
print("Generating predictions using fitted model for abstract data")
predicted_values2 = clf_random_forest_abstract.predict_proba(testing_vector_abstract_representation)

# Make prediction based on the title of job posting using XGBoost
clf_xgb_title = xgb.XGBClassifier(clf.best_params_)
print("SUBMISSION: Fitting XGBoost model to complete title training data")
clf_xgb_title.fit(training_vector_title_representation,
                  training_data_prediction_col_values)
print("SUBMISSION: Generating XGBoost predictions using fitted model for title data")
xgb_predicted_values_from_title_features_model = clf_xgb_title.predict_proba(testing_vector_title_representation)

# Make prediction based on the abstract of the job posting using XGBoost
clf_xgb_abstract = xgb.XGBClassifier(clf.best_params_)
print("SUBMISSION: Fitting XGBoost model to complete abstract training data")
clf_xgb_abstract.fit(training_vector_abstract_representation, training_data_prediction_col_values)
print("SUBMISSION: Generating XGBoost predictions using fitted model for abstract data")
xgb_predicted_values_from_abstract_features_model = clf_xgb_abstract.predict_proba(
    testing_vector_abstract_representation)

predictedTrainingValues1 = clf_random_forest_title.predict_proba()

count = -1

# Lack of time, equal weights for now. maybe use proper stacking next time.
ensembled_prediction = [np.mean(x[0][1] + x[1][1] + x[2][1] + x[3][1])
                        for x in zip(predicted_values1, predicted_values2,
                                     xgb_predicted_values_from_title_features_model,
                                     xgb_predicted_values_from_abstract_features_model)]

# write predictions to console and file
f = open(OUTPUT_FILE, 'w')
# jobIDsAndPredictions = zip(all_job_ids_from_testing_data, predictedValues)
jobIDsAndPredictions = zip(all_job_ids_from_testing_data, ensembled_prediction)
f.write('job_id,hat' + '\n')
for jobID, titlePredictionValue in jobIDsAndPredictions:
    writeString = str(str(jobID) + ',' + str(titlePredictionValue))
    f.write(writeString + '\n')
