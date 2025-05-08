import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from ml_pipeline import MLPipeline, plot_confusion_matrix, plot_top_features
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_whitman_data(base_dir: str) -> pd.DataFrame:
    """
    Load data from Whitman News directory structure.
    
    Args:
        base_dir: Path to the base directory containing category subdirectories.
        
    Returns:
        DataFrame with text content and category labels.
        
    Raises:
        FileNotFoundError: If the base directory doesn't exist.
        Exception: For other errors during data loading.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
        
    data = []
    labels = []
    
    try:
        # Process each category directory
        for category in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category)
            if os.path.isdir(category_path):
                # Process each text file in the category
                logger.info(f"Processing category: {category}")
                file_count = 0
                for file in os.listdir(category_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(category_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            data.append(content)
                            labels.append(category)
                            file_count += 1
                logger.info(f"Processed {file_count} files in category '{category}'")
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': data,
            'category': labels
        })
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    """
    Main function to run the Whitman News processing and classification pipeline.
    
    Workflow:
    1. Load data from Whitman_News directory
    2. Save data to CSV
    3. Run ML pipeline for text classification
    4. Display and visualize results
    """
    try:
        # Load data
        logger.info("Loading Whitman News data...")
        df = load_whitman_data('Whitman_News')
        logger.info(f"Loaded {len(df)} documents across {df['category'].nunique()} categories")
        
        # Save to CSV for the pipeline
        output_csv = 'whitman_news_data.csv'
        df.to_csv(output_csv, index=False)
        logger.info(f"Data saved to {output_csv}")
        
        # Initialize and run pipeline
        logger.info("Running ML pipeline...")
        pipeline = MLPipeline(max_features=1000)
        results = pipeline.run_pipeline(
            data_path=output_csv,
            text_column='text',
            label_column='category',
            test_size=0.15,  # Reduce test size to improve accuracy (more training data)
            random_state=42  # Fixed random state for reproducibility
        )
        
        # Print results
        print("\nClassification Report:")
        # Format the DataFrame to display percentages with % symbol
        df_report = pd.DataFrame(results['classification_report']).transpose()
        
        # Fix support values (divide by 100 since they were multiplied in the evaluation step)
        if 'support' in df_report.columns:
            df_report['support'] = df_report['support'] / 100.0
            df_report['support'] = df_report['support'].apply(lambda x: int(x) if not pd.isna(x) else x)
        
        # Format percentage columns
        for col in df_report.columns:
            if col != 'support':
                df_report[col] = df_report[col].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
                
        print(df_report)
        
        # Create output directory for visualizations if it doesn't exist
        viz_dir = 'visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            logger.info(f"Created visualizations directory: {viz_dir}")
        
        # Additional visualizations
        logger.info("Generating data visualizations...")
        
        # 1. Visualize document length distribution
        doc_length_path = os.path.join(viz_dir, 'document_length_distribution.png')
        visualize_document_lengths(df['text'], doc_length_path)
        
        # 2. Visualize class distribution
        class_dist_path = os.path.join(viz_dir, 'class_distribution.png')
        visualize_class_distribution(df['category'], class_dist_path)
        
        # 3. Generate word clouds for each category
        generate_category_wordclouds(df, 'text', 'category', viz_dir)
        
        # 4. Plot and save confusion matrix
        confusion_matrix_path = os.path.join(viz_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            results['confusion_matrix'],
            classes=pipeline.preprocessor.label_encoder.classes_,
            save_path=confusion_matrix_path
        )
        
        # 5. Plot and save top features
        try:
            # Get and plot top features
            logger.info("Generating top features visualization...")
            top_features = pipeline.feature_extractor.get_top_features(n_features=10)
            top_features_path = os.path.join(viz_dir, 'top_features.png')
            plot_top_features(top_features, save_path=top_features_path)
            logger.info("Top features visualization complete")
            
            # Also display the top features in text format
            print("\nTop 10 Features by TF-IDF Score:")
            for feature, score in sorted(top_features.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature:<30} {score:.4f}")
        except Exception as e:
            logger.warning(f"Could not plot top features: {str(e)}")
            logger.warning(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            
        return results
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def visualize_document_lengths(texts, save_path=None):
    """
    Visualize the distribution of document lengths.
    
    Args:
        texts: Series of text documents
        save_path: Path to save the visualization
    """
    # Calculate document lengths (in words)
    doc_lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(doc_lengths, bins=20, kde=True)
    plt.title('Document Length Distribution (words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.axvline(x=np.mean(doc_lengths), color='r', linestyle='--', 
                label=f'Mean: {np.mean(doc_lengths):.1f} words')
    plt.axvline(x=np.median(doc_lengths), color='g', linestyle='--', 
                label=f'Median: {np.median(doc_lengths):.1f} words')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Document length distribution saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_class_distribution(categories, save_path=None):
    """
    Visualize the distribution of document categories.
    
    Args:
        categories: Series of document categories
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    counts = categories.value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts.values):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Class distribution saved to {save_path}")
    else:
        plt.show()
    plt.close()

def generate_category_wordclouds(df, text_col, cat_col, save_dir):
    """
    Generate word clouds for each document category.
    
    Args:
        df: DataFrame with text and category columns
        text_col: Name of column containing document text
        cat_col: Name of column containing category labels
        save_dir: Directory to save wordcloud images
    """
    # Create a word cloud for each category
    for category in df[cat_col].unique():
        # Combine all text in this category
        category_text = ' '.join(df[df[cat_col] == category][text_col])
        
        # Generate wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(category_text)
        
        # Save the wordcloud
        save_path = os.path.join(save_dir, f'wordcloud_{category}.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud: {category}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Word cloud for category '{category}' saved to {save_path}")

if __name__ == "__main__":
    try:
        main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")