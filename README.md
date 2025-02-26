# Granite TimeSeries Forecasting Tool

Welcome to the Granite TimeSeries Forecasting Tool – an interactive, lightweight forecasting platform based on IBM’s Granite-TimeSeries-TTM model family. This repository contains code for zero-shot forecasting, few-shot fine-tuning (including channel-mix finetuning), and M4 hourly evaluation experiments. The tool is fully containerized and deployable on Hugging Face Spaces, GitHub, or your preferred cloud environment.


<p align="center" width="100%">
<img src="assets/dashboard image new.png" width="600">
</p>



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Deployment](#docker-deployment)
  - [Deploying on Hugging Face Spaces](#deploying-on-hugging-face-spaces)
- [Usage](#usage)
  - [Launching the App](#launching-the-app)
  - [Interactive Modes](#interactive-modes)
- [Data Preparation](#data-preparation)
- [Setting Parameters](#setting-parameters)
- [Interpreting Results](#interpreting-results)
- [Example Notebooks & Code Snippets](#example-notebooks--code-snippets)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Granite TimeSeries Forecasting Tool is an end-to-end solution for time-series forecasting. It leverages tiny pre-trained models (TTMs) for efficient forecasting with both zero-shot and few-shot learning strategies. The tool can:

- Process various time-series datasets.
- Support exogenous/control variables and static categorical features.
- Run advanced experiments such as channel-mix finetuning and rolling forecasts.
- Provide interactive visualizations to inspect predictions.

## Features

- **Zero-shot Evaluation:** Directly apply the pre-trained model on your data.
- **Few-shot Finetuning:** Fine-tune the pre-trained model with a small fraction of your data.
- **Channel-Mix Finetuning:** Leverage conditional (exogenous) features for enhanced performance.
- **M4 Hourly Example:** Replicate experiments on the M4 hourly dataset.
- **Interactive Dashboard:** Use Streamlit to upload data, set parameters, and view interactive results.
- **Dockerized Deployment:** Easily deploy the app via Docker, GitHub, or Hugging Face Spaces.

## Installation

### Local Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/granite-forecasting-tool.git
   cd granite-forecasting-tool

   ```

2. **Install Dependencies:**

We recommend using a virtual environment. Then run:

```bash
pip install -r requirements.txt
```

3. **Run the Application:**

Launch the Streamlit dashboard with:

```bash
streamlit run app.py
```

### Docker Deployment

To build and run the Docker container locally:

1. **Build the Docker Image:**

   ```bash
   docker build -t forecasting-app .

   ```

2. **Run the Container:**

   ```bash
   docker run -p 8501:8501 forecasting-app
   ```

Then open your browser and navigate to http://localhost:8501.

## Usage

### Launching the App

Once running, the app provides an interactive interface where you can:

- Choose between different evaluation modes:
  - **Zero-shot Evaluation**
  - **Channel-Mix Finetuning Example**
  - **M4 Hourly Example**
- Upload your CSV file (or use the default ETTh1 dataset).
- Select target columns, exogenous/control variables, and adjust advanced options (e.g., rolling forecast extension).

### Interactive Modes

**Zero-shot Evaluation**

- **Data Upload:** Upload your time-series CSV file. The file must include a date column.
- **Parameter Setting:**
  - Set context and prediction lengths.
  - Select target columns and optionally exogenous/control columns.
  - Choose a forecast index for dynamic plotting.
- **Run Evaluation:** Click “Run Zero-shot Evaluation” to view evaluation metrics and interactive plots.

**Channel-Mix Finetuning Example**

- This mode demonstrates how to fine-tune the model using channel-mix finetuning on a bike-sharing dataset.
- The tool automatically loads the dataset, sets up the conditional features, freezes the model backbone, and then fine-tunes the model.
- Evaluation metrics and static plots are provided after fine-tuning.

**M4 Hourly Example**

- This mode reproduces experiments on the M4 hourly dataset.
- The tool loads a sample M4 dataset (or a placeholder), evaluates the model in zero-shot mode, and displays results.
- Fine-tuning on M4 can be integrated similarly.

### Data Preparation

Your data must meet the following requirements:

- **CSV Format:** Data should be in CSV format.
- **Timestamp Column:** The CSV must include a column named date (or similar) that is parseable as a date.
- **Target Columns:** These columns contain the time-series values you wish to forecast.
- **Optional Exogenous/Control Variables:** Additional columns used for conditional features.
- **Optional Static Categorical Features:** Columns that represent static characteristics of the time series.

#### Example Data Format

```csv
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT,exog1,exog2
2020-01-01 00:00,100,120,130,140,150,160,170,0.5,low
2020-01-01 01:00,110,125,135,145,155,165,175,0.6,medium
...

```

### Setting Parameters

The following parameters are available for configuration through the dashboard:

- **Context Length:** Number of past time points used as input.
- **Prediction Length:** Number of future time points to forecast.
- **Batch Size:** Batch size used during model evaluation or fine-tuning.
- **Rolling Forecast Extension:** Additional steps for rolling forecasts.
- **Exogenous/Control Variables:** Choose additional columns to be used as conditional features.
- **Static Categorical Features:** (If implemented) Select static features.
- **Evaluation Mode:** Select between zero-shot, channel-mix finetuning, or M4 hourly examples.

### Interpreting Results

After running an evaluation, you will see:

- **Evaluation Metrics:** Metrics such as MSE, MAE, and others, displayed in JSON format.
- **Interactive Plot:** A Plotly chart overlays actual values with the forecast for a selected index. You can zoom, pan, and hover over the chart for detailed information.
- **Static Plots:** Additional static plots (PNG images) are generated and displayed below the interactive plot.
- **Console Logs:** For advanced users, additional logs and printed outputs (e.g., parameter counts) are shown in the terminal or Streamlit’s log panel.

## Contributing

We welcome contributions! Please review our CONTRIBUTING.md file for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the Apache 2.0 License – see the LICENSE file for details.
