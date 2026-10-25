
import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from xgboost import XGBRegressor


# Initialize Flask application
superkart_api = Flask("SuperKart")


# Deployment artifact paths
BASE_DIRECTORY = os.path.dirname(__file__)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIRECTORY,
    "superkart_preprocessor.joblib"
)

MODEL_PATH = os.path.join(
    BASE_DIRECTORY,
    "superkart_xgb_model.json"
)


# Load preprocessing transformer
preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# Load XGBoost through its native model format
model = XGBRegressor()

model.load_model(
    MODEL_PATH
)


EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


VALID_CATEGORIES = {
    "Product_Sugar_Content": [
        "Low Sugar",
        "Regular",
        "No Sugar"
    ],

    "Store_Size": [
        "Small",
        "Medium",
        "High"
    ],

    "Store_Location_City_Type": [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ],

    "Store_Type": [
        "Departmental Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
        "Food Mart"
    ],

    "Product_Id_char": [
        "FD",
        "DR",
        "NC"
    ],

    "Product_Type_Category": [
        "Perishables",
        "Non Perishables"
    ]
}


def validate_input(input_data):
    """
    Validate and order incoming model features.
    """

    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    input_data = input_data[
        EXPECTED_FEATURES
    ].copy()

    numeric_columns = [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Age_Years"
    ]

    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce"
        )

    if input_data[numeric_columns].isnull().any().any():
        raise ValueError(
            "One or more numerical inputs are invalid."
        )

    if (input_data["Product_Weight"] <= 0).any():
        raise ValueError(
            "Product_Weight must be greater than zero."
        )

    if (input_data["Product_MRP"] <= 0).any():
        raise ValueError(
            "Product_MRP must be greater than zero."
        )

    if (
        (input_data["Product_Allocated_Area"] < 0).any()
        or
        (input_data["Product_Allocated_Area"] > 1).any()
    ):
        raise ValueError(
            "Product_Allocated_Area must be between 0 and 1."
        )

    if (input_data["Store_Age_Years"] < 0).any():
        raise ValueError(
            "Store_Age_Years cannot be negative."
        )

    for column, allowed_values in VALID_CATEGORIES.items():

        invalid_values = (
            input_data.loc[
                ~input_data[column].isin(allowed_values),
                column
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if invalid_values:
            raise ValueError(
                f"Invalid values in {column}: "
                f"{invalid_values}. "
                f"Allowed values are {allowed_values}."
            )

    return input_data


def generate_predictions(input_data):
    """
    Apply preprocessing and native XGBoost inference.
    """

    validated_data = validate_input(
        input_data
    )

    transformed_data = preprocessor.transform(
        validated_data
    )

    predictions = model.predict(
        transformed_data
    )

    return predictions


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.post("/v1/predict")
def predict_sales():

    try:
        request_data = request.get_json()

        if request_data is None:
            return jsonify(
                {"error": "No JSON input was provided."}
            ), 400

        input_data = pd.DataFrame(
            [request_data]
        )

        prediction = generate_predictions(
            input_data
        )[0]

        return jsonify(
            {
                "Sales": round(
                    float(prediction),
                    2
                )
            }
        )

    except ValueError as error:
        return jsonify(
            {"error": str(error)}
        ), 400

    except Exception as error:
        return jsonify(
            {
                "error":
                f"Prediction failed: {str(error)}"
            }
        ), 500


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():

    try:
        if "file" not in request.files:
            return jsonify(
                {"error": "No CSV file was uploaded."}
            ), 400

        uploaded_file = request.files["file"]

        input_data = pd.read_csv(
            uploaded_file
        )

        predictions = generate_predictions(
            input_data
        )

        output = {
            str(index): round(
                float(prediction),
                2
            )
            for index, prediction
            in enumerate(predictions)
        }

        return jsonify(output)

    except ValueError as error:
        return jsonify(
            {"error": str(error)}
        ), 400

    except Exception as error:
        return jsonify(
            {
                "error":
                f"Batch prediction failed: {str(error)}"
            }
        ), 500


if __name__ == "__main__":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
