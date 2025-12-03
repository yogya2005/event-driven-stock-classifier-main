# Event-Driven Stock Impact Classifier - Project Summary

## Project Overview

A machine learning system that predicts stock price reactions to news events using three-way classification: Positive (+1), Neutral (0), or Negative (-1) based on next-day price movements exceeding a 1.5% threshold.

**Models Implemented**: Logistic Regression, Random Forest, XGBoost

## Dataset Construction

### Data Source
- FNSPID dataset from HuggingFace (Financial News and Stock Price Integration Dataset)
- Filtered to 2020 data: 20,000 rows sampled from 201,557 available
- Stock data retrieved from yfinance API

### Final Dataset Statistics
- **Total Examples**: 13,568 labeled instances
- **Unique Stocks**: 2,391 (out of 3,213 attempted, 74% success rate)
- **Date Range**: January - June 2020
- **Label Distribution**:
  - Negative (-1): 4,462 (32.9%)
  - Neutral (0): 4,949 (36.5%)
  - Positive (+1): 4,157 (30.6%)

### Data Processing Pipeline
1. Downloaded and cleaned news articles
2. Retrieved historical stock prices and metadata
3. Calculated next-day returns
4. Applied 1.5% threshold for labeling
5. Filtered examples with missing data

## Feature Engineering

### Feature Composition (113 total features)
1. **TF-IDF Features (100)**: Text vectorization of article titles
   - Max features: 100
   - N-grams: unigrams and bigrams
   - Stop words removed

2. **Sector Features (11)**: One-hot encoded stock sectors
   - Technology, Healthcare, Financial Services, etc.

3. **Continuous Features (2)**: Normalized stock characteristics
   - Market capitalization
   - Beta (volatility measure)

### Data Split
- Training set: 10,854 examples (80%)
- Test set: 2,714 examples (20%)
- Stratified sampling to maintain class balance

## Model Performance

### Results Summary

| Model | Test Accuracy | Improvement over Baseline |
|-------|--------------|---------------------------|
| Random Baseline | 33.33% | - |
| Logistic Regression | 42.23% | +8.90 pp |
| Random Forest | 43.81% | +10.48 pp |
| XGBoost | 43.33% | +10.00 pp |

**Best Model**: Random Forest (43.81% accuracy)

### Training Efficiency
- Logistic Regression: 0.11 seconds
- Random Forest: 0.15 seconds
- XGBoost: 0.53 seconds

### Model Analysis
All three models significantly outperform the random baseline, achieving the target accuracy range of 40-60%. Models demonstrate relatively balanced precision and recall across classes, with a slight tendency to favor neutral predictions.

## Technical Implementation

### Task Breakdown
1. **Task 1 - Data Loading**: Load and filter FNSPID dataset
2. **Task 2 - Enrichment**: Download stock data and calculate returns
3. **Task 3 - Feature Engineering**: Create combined feature matrix
4. **Task 4 - Model Training**: Train and evaluate three models
5. **Task 5 - Visualization**: Generate performance charts

### Key Technologies
- **Data Processing**: pandas, numpy
- **Text Features**: scikit-learn TfidfVectorizer
- **Models**: scikit-learn (LR, RF), XGBoost
- **Visualization**: matplotlib, seaborn
- **Financial Data**: yfinance API

## Challenges and Solutions

### Data Availability
- **Challenge**: 26% of stocks failed to download (delisted, incorrect symbols, or missing data)
- **Solution**: Continued with available stocks, filtered incomplete records

### Missing Metadata
- **Challenge**: 15% of examples lacked complete sector/market cap/beta information
- **Solution**: Removed incomplete examples to maintain data quality

### Date Handling
- **Challenge**: News published on weekends/holidays have no corresponding trading data
- **Solution**: Matched to next available trading day for return calculation

### Class Imbalance
- **Challenge**: Initial threshold resulted in excessive neutral classifications
- **Solution**: Adjusted threshold to 1.5% to achieve better class balance

## Generated Artifacts

### Data Files
- `fnspid_2020_cleaned.csv`: Cleaned news data
- `stock_news_labeled.csv`: Labeled dataset with features
- `train_test_split.npz`: Train/test split arrays

### Visualizations
- `model_comparison.png`: Bar chart of model accuracies
- `confusion_matrices.png`: Confusion matrices for all models
- `label_distribution.png`: Dataset class distribution
- `dataset_sample.png`: Sample data table

## Key Findings

### Dataset Characteristics
- Balanced three-way classification achieved through threshold tuning
- Coverage of 11 distinct sectors
- Data represents first half of 2020 (volatile market period)

### Model Insights
- All models achieve similar performance (42-44% accuracy)
- Random Forest slightly outperforms others
- Models predict neutral movements most accurately (highest recall)
- Positive movements are hardest to predict (lowest recall)

### Practical Implications
- 10 percentage point improvement over random baseline demonstrates predictive signal
- Text features from article titles contribute meaningful information
- Stock characteristics (sector, market cap, beta) add predictive value

## Future Improvements

### Data Enhancement
- Expand to full FNSPID dataset (13+ million rows, multiple years)
- Include full article text instead of titles only
- Add temporal features (time of day, day of week)
- Incorporate sentiment analysis

### Feature Engineering
- Advanced NLP techniques (word embeddings, transformers)
- Technical indicators (momentum, volatility)
- Market-wide features (sector performance, VIX)
- Interaction features between text and numerical data

### Model Development
- Hyperparameter tuning via grid search
- Ensemble methods combining multiple models
- Neural network architectures (LSTM, BERT-based)
- Class-weighted training to address remaining imbalance

### Evaluation
- Time-series cross-validation for temporal robustness
- Out-of-sample testing on different time periods
- Trading strategy simulation for practical validation
- Per-sector model performance analysis

## Success Criteria Achieved

- **Dataset**: Created 13,568 labeled examples (target: 10,000+)
- **Features**: Implemented multi-source feature engineering (text + numerical)
- **Models**: Trained and evaluated 3 model types
- **Performance**: Achieved 40-60% target accuracy range
- **Visualizations**: Generated publication-ready figures
- **Documentation**: Complete codebase with reproducible pipeline

## Project Status

**Completion**: All tasks completed successfully
**Runtime**: Approximately 15-20 minutes for full pipeline
**Best Accuracy**: 43.81% (Random Forest)
**Target Met**: Yes (40-60% range)

## Reproducibility

All scripts are organized in the `scripts/` directory and can be run sequentially using `run_full_pipeline.py`. The pipeline saves intermediate outputs, allowing for resumption if interrupted. See README_FULL_PIPELINE.md for detailed execution instructions.

## Conclusion

This project successfully demonstrates that stock price reactions to news events can be predicted with accuracy significantly above random chance. The combination of text features from article titles and stock characteristics provides a strong baseline for further development. The modular pipeline design facilitates experimentation with enhanced features and models.
