
# ============================================================
# SUPERKART FLASK BACKEND
# ============================================================

import os
import numpy as np
import joblib
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify
)


# ============================================================
# INITIALIZE FLASK APPLICATION
# ============================================================

superkart_api = Flask("SuperKart")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# The model is stored in the same folder as app.py

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_model.joblib"
)


# Load the complete serialized pipeline:
# OneHotEncoder + numerical passthrough + Tuned XGBoost

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

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


# ============================================================
# VALID CATEGORICAL VALUES
# ============================================================

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


# ============================================================
# INPUT VALIDATION FUNCTION
# ============================================================

def validate_input(input_data):
    """
    Validate the model input before generating predictions.
    """

    # --------------------------------------------------------
    # Confirm that all required columns exist
    # --------------------------------------------------------

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


    # Keep model features in the exact expected order

    input_data = input_data[
        EXPECTED_FEATURES
    ].copy()


    # --------------------------------------------------------
    # Validate numerical columns
    # --------------------------------------------------------

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

        bad_columns = (
            input_data[numeric_columns]
            .columns[
                input_data[numeric_columns]
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Invalid or missing numerical values in: "
            + ", ".join(bad_columns)
        )


    # --------------------------------------------------------
    # Numerical range checks
    # --------------------------------------------------------

    if (
        input_data["Product_Weight"] <= 0
    ).any():

        raise ValueError(
            "Product_Weight must be greater than zero."
        )


    if (
        input_data["Product_MRP"] <= 0
    ).any():

        raise ValueError(
            "Product_MRP must be greater than zero."
        )


    if (
        (
            input_data[
                "Product_Allocated_Area"
            ] < 0
        ).any()
        or
        (
            input_data[
                "Product_Allocated_Area"
            ] > 1
        ).any()
    ):

        raise ValueError(
            "Product_Allocated_Area must be "
            "between 0 and 1."
        )


    if (
        input_data["Store_Age_Years"] < 0
    ).any():

        raise ValueError(
            "Store_Age_Years cannot be negative."
        )


    # --------------------------------------------------------
    # Validate categorical values
    # --------------------------------------------------------

    for column, valid_values in (
        VALID_CATEGORIES.items()
    ):

        invalid_values = (
            input_data.loc[
                ~input_data[column].isin(
                    valid_values
                ),
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
                f"Allowed values are "
                f"{valid_values}."
            )


    return input_data



# ============================================================
# HOME ROUTE
#
# Template requirement:
# GET /
# ============================================================

@superkart_api.get("/")
def home():

    return "Welcome to the SuperKart System"



# ============================================================
# SINGLE-PREDICTION ENDPOINT
#
# Template requirement:
# POST /v1/predict
# ============================================================

@superkart_api.post("/v1/predict")
def predict_sales():

    try:

        # ----------------------------------------------------
        # Get JSON request
        # ----------------------------------------------------

        data = request.get_json()


        if data is None:

            return jsonify(
                {
                    "error":
                    "No JSON input was provided."
                }
            ), 400


        # ----------------------------------------------------
        # Extract the exact 10 model features
        # ----------------------------------------------------

        sample = {

            "Product_Weight":
                data["Product_Weight"],

            "Product_Sugar_Content":
                data["Product_Sugar_Content"],

            "Product_Allocated_Area":
                data["Product_Allocated_Area"],

            "Product_MRP":
                data["Product_MRP"],

            "Store_Size":
                data["Store_Size"],

            "Store_Location_City_Type":
                data[
                    "Store_Location_City_Type"
                ],

            "Store_Type":
                data["Store_Type"],

            "Product_Id_char":
                data["Product_Id_char"],

            "Store_Age_Years":
                data["Store_Age_Years"],

            "Product_Type_Category":
                data[
                    "Product_Type_Category"
                ]
        }


        # ----------------------------------------------------
        # Convert JSON record to DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [sample]
        )


        # Validate incoming data

        input_data = validate_input(
            input_data
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        ).tolist()[0]


        # Sales revenue should not be negative

        prediction = max(
            float(prediction),
            0.0
        )


        # ----------------------------------------------------
        # Return prediction as JSON
        # ----------------------------------------------------

        return jsonify(
            {
                "Sales": round(
                    prediction,
                    2
                )
            }
        )


    except KeyError as error:

        return jsonify(
            {
                "error":
                f"Missing required field: "
                f"{str(error)}"
            }
        ), 400


    except ValueError as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 400


    except Exception as error:

        return jsonify(
            {
                "error":
                f"Prediction failed: "
                f"{str(error)}"
            }
        ), 500



# ============================================================
# BATCH-PREDICTION ENDPOINT
#
# Template requirement:
# POST /v1/predictbatch
#
# Expected input:
# Batch_Data_SuperKart.csv
# ============================================================

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():

    try:

        # ----------------------------------------------------
        # Check for uploaded file
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify(
                {
                    "error":
                    "No CSV file was uploaded."
                }
            ), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify(
                {
                    "error":
                    "The uploaded file has "
                    "no filename."
                }
            ), 400


        if not file.filename.lower().endswith(
            ".csv"
        ):

            return jsonify(
                {
                    "error":
                    "Only CSV files are supported."
                }
            ), 400


        # ----------------------------------------------------
        # Read uploaded CSV
        # ----------------------------------------------------

        input_data = pd.read_csv(
            file
        )


        # ----------------------------------------------------
        # Validate batch data
        # ----------------------------------------------------

        input_data = validate_input(
            input_data
        )


        # ----------------------------------------------------
        # Generate batch predictions
        # ----------------------------------------------------

        predictions = model.predict(
            input_data
        ).tolist()


        # Do not permit negative sales predictions

        predictions = [
            max(float(prediction), 0.0)
            for prediction in predictions
        ]


        # ----------------------------------------------------
        # Template-compatible output
        #
        # {
        #   "0": 2895.38,
        #   "1": ...,
        #   ...
        # }
        # ----------------------------------------------------

        output_dict = {

            str(i): round(
                prediction,
                2
            )

            for i, prediction
            in enumerate(predictions)

        }


        return jsonify(
            output_dict
        )


    except ValueError as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 400


    except Exception as error:

        return jsonify(
            {
                "error":
                f"Batch prediction failed: "
                f"{str(error)}"
            }
        ), 500



# ============================================================
# RUN FLASK APPLICATION
#
# During Docker deployment Gunicorn starts the application.
# This block supports direct local execution.
# ============================================================

if __name__ == "__main__":

    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
