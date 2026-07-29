"""
create_dataset.py

Generates and saves the Iris dataset locally for tracking with DVC.
Loads data using scikit-learn, converts it to a pandas DataFrame,
and outputs the CSV file into the data/ directory.
"""

import os
import yaml
import pandas as pd
from sklearn.datasets import load_iris


def main():
    """
    Main function to load Iris dataset from sklearn and save it to data/iris.csv.
    """
    # Load parameters from params.yaml if available
    params_path = "params.yaml"
    random_state = 42
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
            random_state = params.get("create_dataset", {}).get("random_state", 42)

    print("Loading Iris dataset from scikit-learn...")
    # Load Iris dataset as DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename columns to standard snake_case naming for clean data schema
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target"
    ]

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Save to data/iris.csv
    output_path = os.path.join(output_dir, "iris.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset successfully created and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")


if __name__ == "__main__":
    main()
