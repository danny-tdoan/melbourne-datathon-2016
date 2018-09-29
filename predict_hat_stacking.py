# code from blend.py (though this is not really traditional blending as described in http://mlwave.com/kaggle-ensembling-guide/
# and used in Kaggle circles. This is essentially stacking (when cv=2) without the part where meta-features are created out
# of predictions made on the whole dataset in one go.

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from feature_words import ALL_IMPORTANT_WORDS

import feature_extractor as fe

OUTPUT_FILE_STACKING = "prediction_stackedlogregression.csv"
OUTPUT_FILE_BAGGING = "prediction_stackedlogregression_bagging.csv"


def create_blend_datasets(X, y, X_submission, label):
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_jobs=8),
            ExtraTreesClassifier(n_jobs=8),
            xgb.XGBClassifier(max_depth=8, n_estimators=400, objective='binary:logistic', learning_rate=0.3),
            LogisticRegression(n_jobs=8)
            ]

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, label, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
            print('Fold ' + str(i) + ' Gini Score ' + str(
                fe.get_gini_score_from_auc_score(roc_auc_score(y_test, y_submission))))

        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    return dataset_blend_train, dataset_blend_test


def blend(dataset_blend_train, dataset_blend_test, y):
    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    return y_submission


def blend_on_training_data_and_calc_metric(dataset_blend_train, training_data_prediction_col_values):
    n_folds = 4
    skf = list(StratifiedKFold(training_data_prediction_col_values, n_folds))
    clf = LogisticRegression()

    for i, (train, test) in enumerate(skf):
        print('Fold ' + str(i))
        X_train = dataset_blend_train[train]
        y_train = training_data_prediction_col_values[train]
        X_test = dataset_blend_train[test]
        y_test = training_data_prediction_col_values[test]

        clf.fit(X_train, y_train)
        predictions_X_test_blending = clf.predict_proba(X_test)[:, 1]
        predictions_X_test_averaging = [sum(x) / len(x) for x in X_test]

        print('Gini score after averaging' + str(
            fe.get_gini_score_from_auc_score(roc_auc_score(y_test, predictions_X_test_averaging))))
        print('Gini score after blending' + str(
            fe.get_gini_score_from_auc_score(roc_auc_score(y_test, predictions_X_test_blending))))


(all_job_ids_from_testing_data,
 trainingVectorTitleRepresentation, trainingVectorAbstractRepresentation, trainingDataPredictionColValues,
 testingVectorTitleRepresentation, testingVectorAbstractRepresentation) = fe.generate_training_and_test_vectors()

n_folds = 2
verbose = True
shuffle = False

X_title, y, X_submission = trainingVectorTitleRepresentation, np.array(
    trainingDataPredictionColValues), testingVectorTitleRepresentation

dataset_blend_train_title, dataset_blend_test_title = create_blend_datasets(trainingVectorTitleRepresentation, y,
                                                                            testingVectorTitleRepresentation,
                                                                            'TitleRepresentation')
dataset_blend_train_abstract, dataset_blend_test_abstract = create_blend_datasets(trainingVectorAbstractRepresentation,
                                                                                  y,
                                                                                  testingVectorAbstractRepresentation,
                                                                                  'AbstractRepresentation')

dataset_blend_train = np.hstack((dataset_blend_train_title, dataset_blend_train_abstract))
dataset_blend_test = np.hstack((dataset_blend_test_title, dataset_blend_test_abstract))

blend_on_training_data_and_calc_metric(dataset_blend_train, np.array(trainingDataPredictionColValues))

ensembled_predicted_values_for_testing_data = blend(dataset_blend_train, dataset_blend_test, y)
averaged_predicted_values_for_testing_data = [sum(x) / len(x) for x in dataset_blend_test]

# write blended predictions to console and file
f = open(OUTPUT_FILE_STACKING, 'w')
job_ids_and_predictions = zip(all_job_ids_from_testing_data, ensembled_predicted_values_for_testing_data)
f.write('job_id,hat' + '\n')
for jobID, titlePredictionValue in job_ids_and_predictions:
    writeString = str(str(jobID) + ',' + str(titlePredictionValue))
    f.write(writeString + '\n')

# write averaged predictions to console and file
f = open(OUTPUT_FILE_BAGGING, 'w')
job_ids_and_predictions = zip(all_job_ids_from_testing_data, averaged_predicted_values_for_testing_data)
f.write('job_id,hat' + '\n')
for jobID, titlePredictionValue in job_ids_and_predictions:
    # print(jobID, predictionValue)
    writeString = str(str(jobID) + ',' + str(titlePredictionValue))
    f.write(writeString + '\n')
