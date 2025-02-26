import os
import math
import tempfile
import warnings
import streamlit as st
import pandas as pd
import torch
import plotly.express as px

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

# For M4 Hourly Example
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

# Suppress warnings and set a reproducible seed
warnings.filterwarnings("ignore")
SEED = 42
set_seed(SEED)

# Default model parameters and output directory
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_PREDICTION_LENGTH = 96
OUT_DIR = "dashboard_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------
# Helper: Interactive Plot
def interactive_plot(actual, forecast, title="Forecast vs Actual"):
    df = pd.DataFrame(
        {"Time": range(len(actual)), "Actual": actual, "Forecast": forecast}
    )
    fig = px.line(df, x="Time", y=["Actual", "Forecast"], title=title)
    return fig


# --------------------------
# Mode 1: Zero-shot Evaluation
def run_zero_shot_forecasting(
    data,
    context_length,
    prediction_length,
    batch_size,
    selected_target_columns,
    selected_conditional_columns,
    rolling_forecast_extension,
    selected_forecast_index,
):
    st.write("### Preparing Data for Forecasting")
    timestamp_column = "date"
    id_columns = []  # Modify if needed.
    # Use selected target columns; default to all columns (except "date") if not provided.
    if not selected_target_columns:
        target_columns = [col for col in data.columns if col != timestamp_column]
    else:
        target_columns = selected_target_columns

    # Incorporate exogenous/control columns.
    conditional_columns = selected_conditional_columns

    # Define column specifiers (if your preprocessor supports static columns, add here)
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": conditional_columns,
    }

    n = len(data)
    split_config = {
        "train": [0, int(n * 0.7)],
        "valid": [int(n * 0.7), int(n * 0.8)],
        "test": [int(n * 0.8), n],
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)
    st.write("Data split into train, validation, and test sets.")

    st.write("### Loading the Pre-trained TTM Model")
    model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    temp_dir = tempfile.mkdtemp()
    training_args = TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=batch_size,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args)

    st.write("### Running Zero-shot Evaluation")
    st.info("Evaluating on the test set...")
    eval_output = trainer.evaluate(dset_test)
    st.write("**Zero-shot Evaluation Metrics:**")
    st.json(eval_output)

    st.write("### Generating Forecast Predictions")
    predictions_dict = trainer.predict(dset_test)
    try:
        predictions_np = predictions_dict.predictions[0]
    except Exception as e:
        st.error("Error extracting predictions: " + str(e))
        return
    st.write("Predictions shape:", predictions_np.shape)

    if rolling_forecast_extension > 0:
        st.write(
            f"### Rolling Forecast Extension: {rolling_forecast_extension} extra steps"
        )
        st.info("Rolling forecast logic can be implemented here.")

    # Interactive plot for a selected forecast index.
    idx = selected_forecast_index
    try:
        # This example assumes dset_test[idx] is a dict with a "target" key; adjust as needed.
        actual = (
            dset_test[idx]["target"]
            if isinstance(dset_test[idx], dict)
            else dset_test[idx][0]
        )
    except Exception:
        actual = predictions_np[idx]  # Fallback if actual is not available.
    fig = interactive_plot(
        actual, predictions_np[idx], title=f"Forecast vs Actual for index {idx}"
    )
    st.plotly_chart(fig)

    # Static plots (generated via plot_predictions)
    plot_dir = os.path.join(OUT_DIR, "zero_shot_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=trainer.model,
            dset=dset_test,
            plot_dir=plot_dir,
            plot_prefix="test_zeroshot",
            indices=[idx],
            channel=0,
        )
    except Exception as e:
        st.error("Error during static plotting: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)


# --------------------------
# Mode 2: Channel-Mix Finetuning Example
def run_channel_mix_finetuning():
    st.write("## Channel-Mix Finetuning Example (Bike Sharing Data)")
    # Load bike sharing dataset
    target_dataset = "bike_sharing"
    DATA_ROOT_PATH = (
        "https://raw.githubusercontent.com/blobibob/bike-sharing-dataset/main/hour.csv"
    )
    timestamp_column = "dteday"
    id_columns = []
    try:
        data = pd.read_csv(DATA_ROOT_PATH, parse_dates=[timestamp_column])
    except Exception as e:
        st.error("Error loading bike sharing dataset: " + str(e))
        return
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    # Adjust timestamps (to add hourly information)
    data[timestamp_column] = data[timestamp_column] + pd.to_timedelta(
        data.groupby(data[timestamp_column].dt.date).cumcount(), unit="h"
    )
    st.write("### Bike Sharing Data Preview")
    st.dataframe(data.head())

    # Define columns: targets and conditional (exogenous) channels
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": ["casual", "registered", "cnt"],
        "conditional_columns": [
            "season",
            "yr",
            "mnth",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ],
    }
    n = len(data)
    split_config = {
        "train": [0, int(n * 0.5)],
        "valid": [int(n * 0.5), int(n * 0.75)],
        "test": [int(n * 0.75), n],
    }
    context_length = 512
    forecast_length = 96

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    train_dataset, valid_dataset, test_dataset = get_datasets(tsp, data, split_config)
    st.write("Data split completed.")

    # For channel-mix finetuning, we use TTM-R1 (as per provided script)
    TTM_MODEL_PATH_CM = "ibm-granite/granite-timeseries-ttm-r1"
    finetune_forecast_model = get_model(
        TTM_MODEL_PATH_CM,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",
        prediction_channel_indices=tsp.prediction_channel_indices,
    )
    st.write(
        "Number of params before freezing backbone:",
        count_parameters(finetune_forecast_model),
    )
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False
    st.write(
        "Number of params after freezing backbone:",
        count_parameters(finetune_forecast_model),
    )

    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
    )
    out_dir = os.path.join(OUT_DIR, target_dataset)
    os.makedirs(out_dir, exist_ok=True)
    finetune_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=1e-5,
    )
    tracking_callback = TrackingCallback()
    finetune_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
    finetune_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])
    st.write("Starting channel-mix finetuning...")
    finetune_trainer.train()
    st.write("Evaluating finetuned model on test set...")
    eval_output = finetune_trainer.evaluate(test_dataset)
    st.write("Few-shot (channel-mix) evaluation metrics:")
    st.json(eval_output)
    # Plot predictions
    plot_dir = os.path.join(out_dir, "channel_mix_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=finetune_trainer.model,
            dset=test_dataset,
            plot_dir=plot_dir,
            plot_prefix="test_channel_mix",
            indices=[0],
            channel=0,
        )
    except Exception as e:
        st.error("Error plotting channel mix predictions: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)


# --------------------------
# Mode 3: M4 Hourly Example
def run_m4_hourly_example():
    st.write("## M4 Hourly Example")
    st.info("This example reproduces a simplified version of the M4 hourly evaluation.")
    # For demonstration, we attempt to load an M4 hourly dataset from a URL.
    # (In practice, you would need to download and prepare the dataset.)
    M4_DATASET_URL = "https://raw.githubusercontent.com/IBM/TSFM-public/main/tsfm_public/notebooks/ETTh1.csv"  # Placeholder URL
    try:
        m4_data = pd.read_csv(M4_DATASET_URL, parse_dates=["date"])
    except Exception as e:
        st.error("Could not load M4 hourly dataset: " + str(e))
        return
    st.write("### M4 Hourly Data Preview")
    st.dataframe(m4_data.head())
    context_length = 512
    forecast_length = 48  # M4 hourly forecast horizon
    timestamp_column = "date"
    id_columns = []
    target_columns = [col for col in m4_data.columns if col != timestamp_column]
    n = len(m4_data)
    split_config = {
        "train": [0, int(n * 0.7)],
        "valid": [int(n * 0.7), int(n * 0.85)],
        "test": [int(n * 0.85), n],
    }
    column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": [],
    }
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    dset_train, dset_valid, dset_test = get_datasets(tsp, m4_data, split_config)
    st.write("Data split completed.")

    # Load model from Hugging Face TTM Model Repository (TTM-V1 for M4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-v1",
        revision="main",
        prediction_filter_length=forecast_length,
    ).to(device)
    st.write("Running zero-shot evaluation on M4 hourly data...")
    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=64,
            report_to="none",
        ),
    )
    eval_output = trainer.evaluate(dset_test)
    st.write("Zero-shot evaluation metrics on M4 hourly:")
    st.json(eval_output)
    plot_dir = os.path.join(OUT_DIR, "m4_hourly", "zero_shot")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_predictions(
            model=trainer.model,
            dset=dset_test,
            plot_dir=plot_dir,
            plot_prefix="m4_zero_shot",
            indices=[0],
            channel=0,
        )
    except Exception as e:
        st.error("Error plotting M4 zero-shot predictions: " + str(e))
        return
    for file in os.listdir(plot_dir):
        if file.endswith(".png"):
            st.image(os.path.join(plot_dir, file), caption=file)
    st.info("Fine-tuning on M4 hourly data can be added similarly.")


# --------------------------
# Main UI
def main():
    st.title("Interactive Time-Series Forecasting Dashboard")
    st.markdown(
        """
        This dashboard lets you run advanced forecasting experiments using the Granite-TimeSeries-TTM model.
        Select one of the modes below:
        - **Zero-shot Evaluation**
        - **Channel-Mix Finetuning Example**
        - **M4 Hourly Example**
        """
    )

    mode = st.selectbox(
        "Select Evaluation Mode",
        options=[
            "Zero-shot Evaluation",
            "Channel-Mix Finetuning Example",
            "M4 Hourly Example",
        ],
    )

    if mode == "Zero-shot Evaluation":
        # Allow user to choose dataset source
        dataset_source = st.radio(
            "Dataset Source", options=["Default (ETTh1)", "Upload CSV"]
        )
        if dataset_source == "Default (ETTh1)":
            DATASET_PATH = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
            try:
                data = pd.read_csv(DATASET_PATH, parse_dates=["date"])
            except Exception as e:
                st.error("Error loading default dataset.")
                return
            st.write("### Default Dataset Preview")
            st.dataframe(data.head())
            selected_target_columns = [
                "HUFL",
                "HULL",
                "MUFL",
                "MULL",
                "LUFL",
                "LULL",
                "OT",
            ]
        else:
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if not uploaded_file:
                st.info("Awaiting CSV file upload.")
                return
            data = pd.read_csv(uploaded_file, parse_dates=["date"])
            st.write("### Uploaded Data Preview")
            st.dataframe(data.head())
            available_columns = [col for col in data.columns if col != "date"]
            selected_target_columns = st.multiselect(
                "Select Target Column(s)",
                options=available_columns,
                default=available_columns,
            )

        # Advanced options
        available_exog = [
            col
            for col in data.columns
            if col not in (["date"] + selected_target_columns)
        ]
        selected_conditional_columns = st.multiselect(
            "Select Exogenous/Control Columns", options=available_exog, default=[]
        )
        rolling_extension = st.number_input(
            "Rolling Forecast Extension (Extra Steps)", value=0, min_value=0, step=1
        )
        forecast_index = st.slider(
            "Select Forecast Index for Plotting",
            min_value=0,
            max_value=len(data) - 1,
            value=0,
        )
        context_length = st.number_input(
            "Context Length", value=DEFAULT_CONTEXT_LENGTH, step=64
        )
        prediction_length = st.number_input(
            "Prediction Length", value=DEFAULT_PREDICTION_LENGTH, step=1
        )
        batch_size = st.number_input("Batch Size", value=64, step=1)
        if st.button("Run Zero-shot Evaluation"):
            with st.spinner("Running zero-shot evaluation..."):
                run_zero_shot_forecasting(
                    data,
                    context_length,
                    prediction_length,
                    batch_size,
                    selected_target_columns,
                    selected_conditional_columns,
                    rolling_extension,
                    forecast_index,
                )

    elif mode == "Channel-Mix Finetuning Example":
        if st.button("Run Channel-Mix Finetuning Example"):
            with st.spinner("Running channel-mix finetuning..."):
                run_channel_mix_finetuning()

    elif mode == "M4 Hourly Example":
        if st.button("Run M4 Hourly Example"):
            with st.spinner("Running M4 hourly example..."):
                run_m4_hourly_example()


if __name__ == "__main__":
    main()
