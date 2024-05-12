from IPython.display import display
import pandas as pd
from apyori import apriori
dataset_a = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Bread'],
    ['Milk', 'Coffee', 'Diapers', 'Beer'],
    ['Diapers', 'Beer', 'Bread', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Diapers', 'Beer']
]

dataset_b = [
    ['Apple', 'Banana', 'Orange'],
    ['Apple', 'Banana'],
    ['Apple', 'Banana', 'Orange', 'Grapes'],
    ['Apple', 'Grapes'],
    ['Banana', 'Orange', 'Grapes']
]
# Function to run Apriori algorithm and print association rules
def run_apriori(dataset, min_support, min_confidence):
    association_rules = apriori(dataset, min_support=min_support, min_confidence=min_confidence)
    results = list(association_rules)
    return results
def display_results(results):
    rows = []
    for result in results:
        row = {
            "Items": ", ".join([item for item in result.items]),
            "Support": round(result.support, 4),
            "Confidence": round(result.ordered_statistics[0].confidence, 4),
            "Lift": round(result.ordered_statistics[0].lift, 4)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
# Evaluation function
def evaluate(dataset, min_support, min_confidence):
    print(f"Evaluating dataset with minimum support={min_support} and minimum confidence={min_confidence}")
    results = run_apriori(dataset, min_support, min_confidence)
    if results:
        df = display_results(results)
        display(df)
    else:
        print("No association rules found.")
# Evaluation
evaluate(dataset_a, min_support=0.5, min_confidence=0.75)
evaluate(dataset_b, min_support=0.6, min_confidence=0.6)
