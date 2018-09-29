import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import custom_csv as customcsv
from constants import *


def generate_training_and_test_vectors():
    """Construct feature vectors from traing and test data.
    Return: two different sets of training/test feature vectors are used for two independent predictors:
        (1) title+job type + salaries + job type + loc, and
        (2) job description (abstract)"""

    print("Reading training data...")
    all_training_rows = customcsv.get_all_training_rows(JOBS_FILE, has_header=True, delimiter="\t",
                                                        label_row_number=COLUMN_NUMBER_TO_PREDICT)
    print("Reading testing data...")
    all_testing_rows = customcsv.get_all_testing_rows(JOBS_FILE, has_header=True, delimiter="\t",
                                                      label_row_number=COLUMN_NUMBER_TO_PREDICT)

    print("Cleaning data...")
    all_abstracts_from_training_data = []
    for trainingRow in all_training_rows:
        all_abstracts_from_training_data.append(trainingRow[ABSTRACT_COLUMN_NUMBER])
    all_cleaned_abstracts_from_training_data = fe.get_clean_text_data(all_abstracts_from_training_data)

    all_abstracts_from_testing_data = []
    for testing_row in all_testing_rows:
        all_abstracts_from_testing_data.append(testing_row[ABSTRACT_COLUMN_NUMBER])
    all_cleaned_abstracts_from_testing_data = fe.get_clean_text_data(all_abstracts_from_testing_data)

    all_titles_from_training_data = []
    for trainingRow in all_training_rows:
        all_titles_from_training_data.append(trainingRow[JOB_TITLE_COLUMN_NUMBER])
    all_cleaned_titles_from_training_data = fe.get_clean_text_data(all_titles_from_training_data)

    all_titles_from_testing_data = []
    for testing_row in all_testing_rows:
        all_titles_from_testing_data.append(testing_row[JOB_TITLE_COLUMN_NUMBER])
    all_cleaned_titles_from_testing_data = fe.get_clean_text_data(all_titles_from_testing_data)

    all_job_types_from_training_data = []
    for trainingRow in all_training_rows:
        all_job_types_from_training_data.append(trainingRow[JOB_TYPE_COLUMN_NUMBER])
    all_cleaned_job_types_from_training_data = fe.get_clean_job_type(all_job_types_from_training_data)

    all_job_types_from_testing_data = []
    for testing_row in all_testing_rows:
        all_job_types_from_testing_data.append(testing_row[JOB_TYPE_COLUMN_NUMBER])
    all_cleaned_job_types_from_testing_data = fe.get_clean_job_type(all_job_types_from_testing_data)

    all_min_salary_from_training_data = []
    for trainingRow in all_training_rows:
        all_min_salary_from_training_data.append(trainingRow[MIN_SAL_COLUMN_NUMBER])
    all_cleaned_min_salary_from_training_data = fe.get_clean_salary(all_min_salary_from_training_data)

    all_min_salary_from_testing_data = []
    for testing_row in all_testing_rows:
        all_min_salary_from_testing_data.append(testing_row[MIN_SAL_COLUMN_NUMBER])
    all_cleaned_min_salary_from_testing_data = fe.get_clean_salary(all_min_salary_from_testing_data)

    all_max_salary_from_training_data = []
    for trainingRow in all_training_rows:
        all_max_salary_from_training_data.append(trainingRow[MAX_SAL_COLUMN_NUMBER])
    all_cleaned_max_salary_from_training_data = fe.get_clean_salary(all_max_salary_from_training_data)

    all_max_salary_from_testing_data = []
    for testing_row in all_testing_rows:
        all_max_salary_from_testing_data.append(testing_row[MAX_SAL_COLUMN_NUMBER])
    all_cleaned_max_salary_from_testing_data = fe.get_clean_salary(all_max_salary_from_testing_data)

    all_raw_loc_from_training_data = []
    for trainingRow in all_training_rows:
        all_raw_loc_from_training_data.append(trainingRow[RAW_LOC_COLUMN_NUMBER])
    all_cleaned_raw_loc_from_training_data = fe.get_clean_text_data(all_raw_loc_from_training_data)

    all_raw_loc_from_testing_data = []
    for testing_row in all_testing_rows:
        all_raw_loc_from_testing_data.append(testing_row[RAW_LOC_COLUMN_NUMBER])
    all_cleaned_raw_loc_from_testing_data = fe.get_clean_text_data(all_raw_loc_from_testing_data)

    all_salary_types = ['below20k', 'above140k', '0']
    for i in range(1, 25):
        all_salary_types.append(str(20000 + i * 5000))

    print("Vectorizing abstract data...")
    training_vector_abstract_representation, training_vectorizer = fe.get_vectorized_representation_using_important_words(
        ALL_IMPORTANT_WORDS, all_cleaned_abstracts_from_training_data)
    print("Finished getting training vector representation")
    testing_vector_abstract_representation, testing_vectorizer = fe.get_vectorized_representation_using_important_words(
        ALL_IMPORTANT_WORDS, all_cleaned_abstracts_from_testing_data)
    print("Finished getting testing vector representation")

    print("Vectorizing title data...")
    training_vector_title_representation, training_vectorizer = fe.get_vectorized_representation_using_important_words(
        ALL_IMPORTANT_WORDS, all_cleaned_titles_from_training_data)
    print("Finished getting training vector representation")
    testing_vector_title_representation, testing_vectorizer = fe.get_vectorized_representation_using_important_words(
        ALL_IMPORTANT_WORDS, all_cleaned_titles_from_testing_data)
    print("Finished getting testing vector representation")

    print("Vectorizing job type...")
    training_vector_job_type_representation, training_vectorizer_title = fe.get_vectorized_representation_of_job_type(
        ALL_JOB_TYPES, all_cleaned_job_types_from_training_data)
    print("Finished getting training vector representation")
    testing_vector_job_type_representation, testing_vectorizer_title = fe.get_vectorized_representation_of_job_type(
        ALL_JOB_TYPES, all_cleaned_job_types_from_testing_data)
    print("Finished getting testing vector representation")

    print("Vectorizing min salary...")
    training_vector_min_sal_representation, training_vectorizer_min_sal = \
        fe.get_vectorized_representation_using_important_words(all_salary_types,
                                                               all_cleaned_min_salary_from_training_data)
    testing_vector_min_sal_representation, testing_vectorizer_min_sal = \
        fe.get_vectorized_representation_using_important_words(all_salary_types,
                                                               all_cleaned_min_salary_from_testing_data)

    print("Vectorizing max salary...")
    training_vector_max_sal_representation, training_vectorizer_max_sal = \
        fe.get_vectorized_representation_using_important_words(all_salary_types,
                                                               all_cleaned_max_salary_from_training_data)
    testing_vector_max_sal_representation, testing_vectorizer_max_sal = \
        fe.get_vectorized_representation_using_important_words(all_salary_types,
                                                               all_cleaned_max_salary_from_testing_data)

    # form the training and testing vectors from the feature
    training_vector_title_representation = \
        np.asarray([np.concatenate((list(x[0]), list(x[1]), list(x[2]), list(x[3]), list(x[4])), axis=0)
                    for x in list(zip(training_vector_title_representation,
                                      training_vector_job_type_representation,
                                      training_vector_min_sal_representation,
                                      training_vector_max_sal_representation,
                                      all_cleaned_raw_loc_from_training_data))])

    testing_vector_title_representation = \
        np.asarray([np.concatenate((list(x[0]), list(x[1]), list(x[2]), list(x[3]), list(x[4])), axis=0)
                    for x in list(zip(testing_vector_title_representation,
                                      testing_vector_job_type_representation,
                                      testing_vector_min_sal_representation,
                                      testing_vector_max_sal_representation,
                                      all_cleaned_raw_loc_from_testing_data))])

    print("Reading column to predict...")
    training_data_prediction_col_values = []
    for trainingRow in all_training_rows:
        training_data_prediction_col_values.append(trainingRow[COLUMN_NUMBER_TO_PREDICT])

    # convert each element to int. roc_auc scoring metric needs this
    training_data_prediction_col_values = list(map(int, training_data_prediction_col_values))

    all_job_ids_from_testing_data = []
    for testing_row in all_testing_rows:
        all_job_ids_from_testing_data.append(testing_row[JOB_ID_COLUMN])

    return (all_job_ids_from_testing_data,
            training_vector_title_representation,
            training_vector_abstract_representation,
            training_data_prediction_col_values,
            testing_vector_title_representation,
            testing_vector_abstract_representation)


def get_clean_text_data(allRows):
    """clean texts of each row, remove numbers, stop words, convert all to lower case and returns a list
    of strings (one for each row)"""

    # remove words with numbers in them
    all_rows_numbers_removed = []
    for row in allRows:
        all_rows_numbers_removed.append(re.sub("[^a-zA-Z]", " ", row))

    # convert all words to lower case
    all_rows_lower_case_letters = []
    for numbersRemovedRow in all_rows_numbers_removed:
        all_rows_lower_case_letters.append(numbersRemovedRow.lower())

    # remove stop words
    stops = set(stopwords.words("english"))
    all_rows_lower_case_stop_words_removed = []
    for lowercaseRow in all_rows_lower_case_letters:
        words = lowercaseRow.split()
        meaningful_words = [w for w in words if not w in stops]
        all_rows_lower_case_stop_words_removed.append(" ".join(meaningful_words))

    return all_rows_lower_case_stop_words_removed


def get_clean_job_type(allRows):
    """clean job types, render all to consistent convention"""

    synonymDict = {}
    synonymDict['fulltime'] = ['full-time']
    synonymDict['parttime'] = ['part-time']

    res = []
    for row in allRows:
        row = row.lower().replace(' ', '')

        # unify the delimiter
        if '/' in row:
            row = row.replace('/', ' ')
        if ',' in row:
            row = row.replace(',', ' ')
        for key, value in synonymDict.items():
            for v in value:
                if v in row:
                    row = row.replace(v, key)
                    break
        res.append(row)

    return res


def get_clean_salary(allRows):
    """clean and categorize salary details"""
    res = []

    # all possible classification of salary
    salary_class = {}
    salary_class[0] = "0"

    # for everything that is below 25k
    salary_class[20000] = "below20k"

    # now the step, the key x groups all salaries which are less than or equal x but greater than x-1
    # from the database, the step is 5k
    for i in range(1, 25):
        salary_class[20000 + i * 5000] = str(20000 + i * 5000)
    salary_class[140001] = "above140k"

    for row in allRows:
        try:
            if row:
                # get the salary value and classify it
                # round down to the neareast 5k
                salary = (int(float(row)) // 5000) * 5000
                if salary <= 20000:
                    res.append(salary_class[20000])
                elif salary > 140000:
                    res.append(salary_class[140001])
                else:
                    res.append(salary_class[salary])
            else:
                res.append("0")
        except ValueError:
            res.append("0")

    return res


def get_vectorized_representation_of_job_type(all_job_types, all_text_vals_in_column):
    """Binary feature vectors of job type"""
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 vocabulary=all_job_types)
    print("Got CountVectorizer")
    all_text_vals_in_column = list(map(lambda x: x.lower(), all_text_vals_in_column))
    word_counts = vectorizer.fit_transform(all_text_vals_in_column).toarray()

    print("Finished fit and transform")
    return word_counts, vectorizer


def get_vectorized_representation_using_important_words(all_important_words, all_text_vals_in_column):
    """Generate feature vectors from the title or abstract based on pre-defined relevant keywords"""
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 vocabulary=all_important_words)

    all_text_vals_in_column = list(map(lambda x: x.lower(), all_text_vals_in_column))
    word_counts = vectorizer.fit_transform(all_text_vals_in_column).toarray()

    print("Finished fit and transform")
    return word_counts, vectorizer