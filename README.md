# How to Run the Event-Driven Stock Classifier

This document provides instructions for running the stock classifier pipeline.

## Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

The project is organized into four main directories:

- **scripts/**: All Python scripts for data processing, training, and evaluation
- **data/**: Input datasets and processed data files
- **models/**: Trained models and feature processors
- **visualizations/**: Generated plots and figures

## Running the Pipeline

### Option 1: Run Full Pipeline (Recommended)

Execute all tasks sequentially:

```bash
cd scripts
python run_full_pipeline.py
```

This will run Tasks 1-5 in order. The pipeline saves progress after each task, so you can resume if interrupted.

### Option 2: Run Individual Tasks

Execute tasks separately:

```bash
cd scripts

# Task 1: Load and filter FNSPID dataset
python task1_load_all_data.py

# Task 2: Enrich with stock data from yfinance
python task2_enrich_all_data.py

# Task 3: Feature engineering
python task3_feature_engineering_all.py

# Task 4: Train models
python task4_train_models_all.py

# Task 5: Create visualizations
python task5_create_visualizations_all.py
```

## Pipeline Tasks Overview

### Task 1: Data Loading
- Downloads FNSPID dataset from HuggingFace
- Cleans and filters news articles
- Output: `../data/fnspid_all_cleaned.csv`

### Task 2: Stock Data Enrichment
- Downloads stock metadata and historical prices using yfinance
- Calculates next-day returns
- Assigns labels: Positive (+1), Neutral (0), Negative (-1) based on 1.5% threshold
- Output: `../data/stock_news_all_labeled.csv`

### Task 3: Feature Engineering
- TF-IDF vectorization of article titles (100 features)
- One-hot encoding of stock sectors (approximately 11 features)
- Normalization of continuous features (market cap, beta)
- Train/test split (80/20)
- Outputs: `../data/train_test_split_all.npz` and feature processors in `../models/`

### Task 4: Model Training
- Trains three models: Logistic Regression, Random Forest, XGBoost
- Evaluates performance with accuracy, classification reports, and confusion matrices
- Outputs: Trained models saved in `../models/`

### Task 5: Visualization
- Creates model comparison charts
- Generates confusion matrices
- Plots label distribution
- Shows dataset samples
- Outputs: All plots saved in `../visualizations/`

## Additional Analysis Scripts

After running the pipeline, you can perform additional analysis:

```bash
cd scripts

# Test for overfitting
python test_overfitting.py

# Check prediction distributions
python check_predictions.py

# Explore dataset structure
python explore_dataset.py
```

## Output Files

### Data Files (`data/`)
- `fnspid_all_cleaned.csv`: Cleaned news data
- `stock_news_all_labeled.csv`: Labeled dataset with features
- `train_test_split_all.npz`: Train/test split arrays

### Model Files (`models/`)
- `logistic_regression_model_all.pkl`: Logistic Regression model
- `xgboost_model_all.pkl`: XGBoost model
- `tfidf_vectorizer_all.pkl`: TF-IDF text vectorizer
- `feature_scaler_all.pkl`: Feature normalizer
- `model_results_all.pkl`: Performance metrics

### Visualization Files (`visualizations/`)
- `model_comparison_all.png`: Model accuracy comparison
- `confusion_matrices_all.png`: Confusion matrices for all models
- `label_distribution_all.png`: Dataset class distribution
- `dataset_sample_all.png`: Sample data table
- `overfitting_analysis_all.png`: Overfitting diagnostics (from test_overfitting.py)

## Key Features

### Progress Caching
Task 2 implements caching to save downloaded stock data. If interrupted, the script will resume from the last saved checkpoint (`stock_data_cache.pkl`).

### Memory Efficiency
The pipeline uses sparse matrices for TF-IDF features and processes data in manageable chunks to minimize memory usage.

### Error Handling
- Missing stocks: Some stocks may fail to download (delisted or invalid symbols). The pipeline continues with available data.
- Weekend/holidays: News published on non-trading days are matched to the next available trading day.

## Troubleshooting

### Task 1: Download Fails
If the HuggingFace download fails, manually download the dataset:
1. Visit https://huggingface.co/datasets/Zihan1004/FNSPID
2. Download `Stock_news/All_external.csv`
3. Update the file path in `task1_load_all_data.py`

### Task 2: Slow Execution
Task 2 is the longest step due to API calls to yfinance. To speed up:
- Reduce the number of stocks processed
- Ensure stable internet connection
- Wait for API rate limits to reset if requests are failing

### Memory Issues
If you encounter out-of-memory errors:
- Close other applications
- Sample the dataset to fewer rows in Task 1
- Process data in smaller batches in Task 2

### Models Not Found
If scripts cannot find model files, ensure you're running from the `scripts/` directory. All paths are relative to the scripts folder.

## Additional Documentation

- **README.md**: Project overview and directory structure
- **PROJECT_SUMMARY.md**: Detailed project results and findings
- **CLAUDE.md**: Step-by-step implementation guide

## Support

For issues or questions, review the terminal output for error messages and check that all dependencies are installed correctly.
