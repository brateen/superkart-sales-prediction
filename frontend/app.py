
# ============================================================
# SUPERKART STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import pandas as pd
import requests


# ============================================================
# BACKEND CONFIGURATION
#
# "backend" is the Docker container name.
# Both backend and frontend containers must run
# on the same Docker network.
# ============================================================

BACKEND_URL = "http://backend:7860"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SuperKart System",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# PAGE TITLE AND DESCRIPTION
# ============================================================

st.title("🛒 SuperKart System")

st.write(
    "Enter the product and store details below "
    "to predict the total product-store sales."
)


# ============================================================
# SINGLE PRODUCT PREDICTION
# ============================================================

st.subheader("Single Product Prediction")


# ------------------------------------------------------------
# PRODUCT INPUTS
# ------------------------------------------------------------

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.01,
    value=12.66,
    step=0.01
)


Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "Regular",
        "No Sugar"
    ]
)


Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.000,
    max_value=1.000,
    value=0.027,
    step=0.001,
    format="%.3f",
    help=(
        "Ratio of allocated product display area "
        "to total store display area."
    )
)


Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.01,
    value=117.08,
    step=0.01
)


# ------------------------------------------------------------
# STORE INPUTS
# ------------------------------------------------------------

Store_Size = st.selectbox(
    "Store Size",
    [
        "Small",
        "Medium",
        "High"
    ],
    index=1
)


Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ],
    index=1
)


# IMPORTANT:
# Only store types present in the SuperKart training data
# and accepted by the Flask backend are included.

Store_Type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Departmental Store",
        "Food Mart"
    ],
    index=1
)


Product_Id_char = st.selectbox(
    "Product ID Character",
    [
        "FD",
        "DR",
        "NC"
    ]
)


Store_Age_Years = st.number_input(
    "Store Age (Years)",
    min_value=0,
    max_value=100,
    value=16,
    step=1,
    help=(
        "The model was trained using Store_Age_Years "
        "= 2025 - Store_Establishment_Year."
    )
)


Product_Type_Category = st.selectbox(
    "Product Type Category",
    [
        "Perishables",
        "Non Perishables"
    ],
    index=1
)


# ============================================================
# CREATE JSON PAYLOAD
# ============================================================

product_data = {

    "Product_Weight":
        float(Product_Weight),

    "Product_Sugar_Content":
        Product_Sugar_Content,

    "Product_Allocated_Area":
        float(Product_Allocated_Area),

    "Product_MRP":
        float(Product_MRP),

    "Store_Size":
        Store_Size,

    "Store_Location_City_Type":
        Store_Location_City_Type,

    "Store_Type":
        Store_Type,

    "Product_Id_char":
        Product_Id_char,

    "Store_Age_Years":
        int(Store_Age_Years),

    "Product_Type_Category":
        Product_Type_Category
}


# ============================================================
# SINGLE PREDICTION
#
# Calls:
# POST http://backend:7860/v1/predict
#
# Expected backend response:
# {"Sales": 2895.38}
# ============================================================

if st.button(
    "Predict",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Generating sales prediction..."
        ):

            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                json=product_data,
                timeout=30
            )


        if response.status_code == 200:

            result = response.json()

            predicted_sales = result["Sales"]


            st.success(
                "Prediction completed successfully!"
            )


            st.metric(
                label=(
                    "Predicted Product Store Sales Total"
                ),
                value=f"{predicted_sales:,.2f}"
            )


            st.write(
                "The estimated product-store sales total "
                f"for the entered combination is "
                f"**{predicted_sales:,.2f}**."
            )


        else:

            try:

                error_result = response.json()

                error_message = error_result.get(
                    "error",
                    str(error_result)
                )

            except Exception:

                error_message = response.text


            st.error(
                "Prediction API returned an error: "
                f"{error_message}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the backend API. "
            "Confirm that the backend container is running "
            "with the name 'backend' and both containers "
            "are connected to the same Docker network."
        )


    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )


    except Exception as error:

        st.error(
            f"Prediction failed: {str(error)}"
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

st.divider()

st.subheader("Batch Prediction")


st.write(
    "Upload **Batch_Data_SuperKart.csv** "
    "to generate predictions for multiple "
    "product-store combinations."
)


# Model features expected by backend

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


uploaded_file = st.file_uploader(
    "Upload Batch_Data_SuperKart.csv",
    type=["csv"]
)


# ============================================================
# VALIDATE AND PREVIEW UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        uploaded_file.seek(0)

        preview_data = pd.read_csv(
            uploaded_file
        )


        st.write(
            f"Records detected: "
            f"**{len(preview_data)}**"
        )


        st.write(
            "Batch Data Preview"
        )


        st.dataframe(
            preview_data,
            use_container_width=True
        )


        missing_columns = [

            column

            for column in EXPECTED_FEATURES

            if column not in preview_data.columns

        ]


        if missing_columns:

            st.error(
                "Uploaded file is missing required columns: "
                + ", ".join(missing_columns)
            )


        else:

            st.success(
                "Batch file schema validation passed."
            )


            # =================================================
            # BATCH PREDICTION BUTTON
            # =================================================

            if st.button(
                "Predict for Batch",
                type="primary",
                use_container_width=True
            ):

                try:

                    # Reset file pointer before HTTP upload

                    uploaded_file.seek(0)


                    files = {

                        "file": (

                            uploaded_file.name,

                            uploaded_file.getvalue(),

                            "text/csv"

                        )
                    }


                    with st.spinner(
                        "Generating batch predictions..."
                    ):

                        response = requests.post(

                            f"{BACKEND_URL}/v1/predictbatch",

                            files=files,

                            timeout=120

                        )


                    # =========================================
                    # PROCESS SUCCESSFUL RESPONSE
                    # =========================================

                    if response.status_code == 200:

                        results = response.json()


                        st.success(
                            "Predictions completed successfully!"
                        )


                        # Backend returns:
                        #
                        # {
                        #   "0": 2768.56,
                        #   "1": 2647.99,
                        #   ...
                        # }


                        if isinstance(
                            results,
                            dict
                        ):

                            prediction_df = pd.DataFrame(

                                list(
                                    results.items()
                                ),

                                columns=[
                                    "Row",
                                    "Predicted_Sales"
                                ]

                            )


                            prediction_df[
                                "Row"
                            ] = (

                                prediction_df[
                                    "Row"
                                ]
                                .astype(int)

                            )


                            prediction_df = (
                                prediction_df
                                .sort_values("Row")
                                .reset_index(drop=True)
                            )


                            # Combine original batch records
                            # with predictions

                            results_df = (
                                preview_data
                                .reset_index(drop=True)
                                .copy()
                            )


                            results_df[
                                "Predicted_Product_Store_Sales_Total"
                            ] = prediction_df[
                                "Predicted_Sales"
                            ].values


                            st.write(
                                "### Batch Prediction Results"
                            )


                            st.dataframe(
                                results_df,
                                use_container_width=True
                            )


                            # =================================
                            # SUMMARY METRICS
                            # =================================

                            col1, col2, col3 = (
                                st.columns(3)
                            )


                            col1.metric(
                                "Predictions",
                                f"{len(results_df):,}"
                            )


                            col2.metric(
                                "Average Predicted Sales",
                                (
                                    f"{results_df['Predicted_Product_Store_Sales_Total'].mean():,.2f}"
                                )
                            )


                            col3.metric(
                                "Maximum Predicted Sales",
                                (
                                    f"{results_df['Predicted_Product_Store_Sales_Total'].max():,.2f}"
                                )
                            )


                            # =================================
                            # DOWNLOAD PREDICTION RESULTS
                            # =================================

                            csv_output = (
                                results_df
                                .to_csv(index=False)
                                .encode("utf-8")
                            )


                            st.download_button(

                                label=(
                                    "Download Prediction Results"
                                ),

                                data=csv_output,

                                file_name=(
                                    "SuperKart_Batch_Predictions.csv"
                                ),

                                mime="text/csv",

                                use_container_width=True

                            )


                        else:

                            st.json(results)


                    # =========================================
                    # BACKEND ERROR RESPONSE
                    # =========================================

                    else:

                        try:

                            error_result = response.json()

                            error_message = (
                                error_result.get(
                                    "error",
                                    str(error_result)
                                )
                            )

                        except Exception:

                            error_message = (
                                response.text
                            )


                        st.error(
                            "Batch prediction API "
                            "returned an error: "
                            f"{error_message}"
                        )


                except requests.exceptions.ConnectionError:

                    st.error(
                        "Unable to connect to the backend API. "
                        "Confirm that the backend container "
                        "is running with the name 'backend' "
                        "and is connected to the same "
                        "Docker network."
                    )


                except requests.exceptions.Timeout:

                    st.error(
                        "The batch prediction request "
                        "timed out."
                    )


                except Exception as error:

                    st.error(
                        f"Batch prediction failed: "
                        f"{str(error)}"
                    )


    except Exception as error:

        st.error(
            "Unable to read the uploaded CSV file: "
            f"{str(error)}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Model Information"
):

    st.write(
        "**Final Model:** Tuned XGBoost"
    )

    st.write(
        "**Test RMSE:** 295.77"
    )

    st.write(
        "**Test MAE:** 122.88"
    )

    st.write(
        "**Test R-squared:** 0.9236"
    )

    st.write(
        "**Test MAPE:** 5.40%"
    )

    st.write(
        "The model pipeline includes categorical "
        "one-hot encoding and the tuned XGBoost "
        "regression model."
    )
