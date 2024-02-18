import os
import pandas as pd
from constants import TASK_NAMES, SELECTIVE_ANNOTATION_METHODS


# Function to parse the accuracy from a result_summary.txt file
def parse_accuracy(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        accuracy = content.split("accuracy is: ")[1].strip()
    return float(accuracy)


rows = []
# Iterate over every combination of task and method
for task in TASK_NAMES:
    for method in SELECTIVE_ANNOTATION_METHODS:
        file_path = os.path.join("outputs", task, method, "result_summary.txt")
        if os.path.exists(file_path):
            accuracy = parse_accuracy(file_path)
            rows.append({"task": task, "method": method, "accuracy": accuracy})
        else:
            print(f"File not found: {file_path}")

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(rows)

# Pivot the DataFrame
pivot_df = df.pivot(index="method", columns="task", values="accuracy")

# Add an 'average' column which is the row average
pivot_df["average"] = pivot_df.mean(axis=1)

# Sort the DataFrame based on the 'average' column in ascending order
pivot_df_sorted = pivot_df.sort_values(by="average")

# Save the sorted pivoted DataFrame to a CSV file
pivot_df_sorted.to_csv("results_summary.csv")

print("Sorted CSV file has been created as 'results_summary.csv'.")
