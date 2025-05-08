# ML Pipeline for Text Classification

A machine learning pipeline for text classification built with scikit-learn, designed to classify Whitman News articles into different categories.

## Features

- Data preprocessing and text cleaning
- Feature extraction using TF-IDF vectorization
- Text classification using Random Forest
- Evaluation metrics and visualizations
- Support for custom datasets

## Visualizations

The pipeline generates several visualizations:
- Document length distribution
- Class distribution
- Word clouds for each category
- Confusion matrix
- Top features by TF-IDF scores

## Requirements

```
pandas>=2.2.3
numpy>=2.2.5
scikit-learn>=1.6.1
matplotlib>=3.10.1
seaborn>=0.13.2
wordcloud>=1.9.4
```

## Usage

1. Place your text data in categorized directories
2. Run the pipeline:

```python
python process_whitman_news.py
```

## Project Structure

- `ml_pipeline.py` - Core machine learning pipeline components
- `process_whitman_news.py` - Data loading and processing for Whitman News
- `Whitman_News/` - Sample dataset directory

## Performance

The model achieves 100% accuracy on the Whitman News dataset.
