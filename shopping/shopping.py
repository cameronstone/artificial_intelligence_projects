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
    print(evidence[0:5])
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
    df = pd.read_csv(filename)

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
        
    def convert_visitor(visitor):
        if visitor == "Returning_Visitor":
            return 1
        else:
            return 0
    
    def convert_weekend(weekend):
        if weekend:
            return 1
        else:
            return 0

    def convert_revenue(revenue):
        if revenue:
            return 1
        else:
            return 0

    df['Month'] = df.Month.apply(lambda row: convert_month(row))
    df['VisitorType'] = df.VisitorType.apply(lambda row: convert_visitor(row))
    df['Weekend'] = df.Weekend.apply(lambda row: convert_weekend(row))
    df['Revenue'] = df.Revenue.apply(lambda row: convert_revenue(row))

    # df.to_csv(r'/Users/cameron/Desktop/artificial_intelligence_projects/shopping/test.csv')

    labels = df.Revenue.values.tolist()
    df = df.drop(columns=['Revenue'])

    # convert column by column to preserve data types (StackOverflow)
    evidence = [df[col].values.tolist() for col in df.columns]
    evidence = list(list(col) for col in zip(*evidence))

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


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
    raise NotImplementedError


if __name__ == "__main__":
    main()
