import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # convert csv to pandas dataframe
    df = pd.read_csv(filename)

    # helper function to assign month integer val
    def convert_month(month):
        if month == "Jan":
            return 0
        elif month == "Feb":
            return 1
        elif month == "Mar":
            return 2
        elif month == "Apr":
            return 3
        elif month == "May":
            return 4
        elif month == "Jun":
            return 5
        elif month == "Jul":
            return 6
        elif month == "Aug":
            return 7
        elif month == "Sep":
            return 8
        elif month == "Oct":
            return 9
        elif month == "Nov":
            return 10
        else:
            return 11

    # helper function to assign visitor integer val
    def convert_visitor(visitor):
        if visitor == "Returning_Visitor":
            return 1
        else:
            return 0

    # helper function to assign weekend integer val
    def convert_weekend(weekend):
        if weekend:
            return 1
        else:
            return 0

    # helper function to assign 'labels' integer val
    def convert_revenue(revenue):
        if revenue:
            return 1
        else:
            return 0

    # convert columns to proper data types using lambda func
    df['Month'] = df.Month.apply(lambda row: convert_month(row))
    df['VisitorType'] = df.VisitorType.apply(lambda row: convert_visitor(row))
    df['Weekend'] = df.Weekend.apply(lambda row: convert_weekend(row))
    df['Revenue'] = df.Revenue.apply(lambda row: convert_revenue(row))

    # create labels list
    labels = df.Revenue.values.tolist()
    # drop from dataframe to create evidence list
    df = df.drop(columns=['Revenue'])

    # convert by column to preserve dtypes (StackOverflow)
    evidence = [df[col].values.tolist() for col in df.columns]
    evidence = list(list(col) for col in zip(*evidence))

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # create KNN model with k=1
    model = KNeighborsClassifier(n_neighbors=1)
    # fit model to training data
    model = model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # count total actual positives/negatives
    total_positive = 0
    total_negative = 0
    # count total correctly identified pos/neg
    correct_pos_prediction = 0
    correct_neg_prediction = 0
    # loop through every indice in labels
    for i in range(len(labels)):
        # identify actual positives
        if labels[i] == 1:
            total_positive += 1
            # identify correctly predicted positives
            if predictions[i] == 1:
                correct_pos_prediction += 1
        # identify actual negatives
        else:
            total_negative += 1
            # identify correctly predicted negatives
            if predictions[i] == 0:
                correct_neg_prediction += 1
    # sensitivity shows true positive rate
    sensitivity = correct_pos_prediction / total_positive
    # specificty shows true negative rate
    specificity = correct_neg_prediction / total_negative
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
