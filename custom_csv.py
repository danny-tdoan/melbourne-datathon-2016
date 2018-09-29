"""Some simple utilities to extract data from csv training and test data. In restropect, this could have been much
more efficient to use pandas"""

import csv

def get_all_rows_for_column (file, columnNumber, has_header, delimiter):
    """Extract data column-wise. Used to extract features (title/salary, etc.) to construct feature vectors"""
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        jobsReader = csv.reader(csvfile, delimiter=delimiter)

        if (has_header):
            next(jobsReader, None)

        allrows = []
        for row in jobsReader:
            allrows.append(row[columnNumber])

    return allrows


def get_all_training_rows(file, has_header, delimiter, label_row_number):
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        jobsReader = csv.reader(csvfile, delimiter=delimiter)

        if (has_header):
            next(jobsReader, None)

        allrows = []
        for row in jobsReader:
            if row[label_row_number]!= '-1':
                allrows.append(row)

    return allrows


def get_all_testing_rows(file, has_header, delimiter, label_row_number):
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        jobsReader = csv.reader(csvfile, delimiter=delimiter)

        if (has_header):
            next(jobsReader, None)

        allrows = []
        for row in jobsReader:
            if row[label_row_number]== '-1':
                allrows.append(row)

    return allrows
