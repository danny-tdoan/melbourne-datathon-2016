JOB_TITLE_COLUMN_NUMBER = 1
ABSTRACT_COLUMN_NUMBER = 9
JOB_TYPE_COLUMN_NUMBER = 8
COLUMN_NUMBER_TO_PREDICT = 11
JOB_ID_COLUMN = 0
MIN_SAL_COLUMN_NUMBER = 6
MAX_SAL_COLUMN_NUMBER = 7
RAW_LOC_COLUMN_NUMBER = 2
ALL_JOB_TYPES = ['fulltime', 'partime', 'contract', 'temp', 'permanent', 'casual', 'vacation', 'temporary', 'employed',
                 'daily', 'hourlyrate', 'attorney', 'fixedterm', 'internship', 'workexperience', 'ongoing',
                 'continuing']

ALL_IMPORTANT_WORDS = list(set(fw.ALL_IMPORTANT_WORDS))  # removing duplicates
JOBS_FILE = "data/jobs_all.csv"  # jobs_small.csv for scaled down inputs
TRAINING_PREDICTION_OUTPUT_FILE = "privated_training_jobs.csv"
OUTPUT_FILE = "prediction.csv"

DELIMITER = '\t'

NUM_CROSS_VALIDATION_FOLDS = 10
