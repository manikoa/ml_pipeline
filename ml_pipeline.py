import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing text data and encoding labels."""
    
    def __init__(self):
        """Initialize the data preprocessor with a label encoder."""
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            DataFrame containing the loaded data.
            
        Raises:
            Exception: If there's an error loading the data.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_text(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess text data by dropping null values and converting text to lowercase.
        
        Args:
            df: DataFrame containing the text data.
            text_column: Name of the column containing text data.
            
        Returns:
            DataFrame with preprocessed text.
            
        Raises:
            Exception: If there's an error in text preprocessing.
        """
        try:
            # Remove null values
            df = df.dropna(subset=[text_column])
            
            # Convert text to lowercase
            df[text_column] = df[text_column].str.lower()
            
            logger.info("Text preprocessing completed")
            return df
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise

    def encode_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Encode categorical labels using LabelEncoder.
        
        Args:
            df: DataFrame containing the labels.
            label_column: Name of the column containing labels.
            
        Returns:
            Tuple containing (DataFrame, encoded labels).
            
        Raises:
            Exception: If there's an error encoding labels.
        """
        try:
            labels = self.label_encoder.fit_transform(df[label_column])
            logger.info(f"Labels encoded successfully. Unique classes: {len(self.label_encoder.classes_)}")
            return df, labels
        except Exception as e:
            logger.error(f"Error encoding labels: {str(e)}")
            raise

class FeatureExtractor:
    """Class for extracting and analyzing text features using TF-IDF."""
    
    def __init__(self, max_features: int = 1000):
        """Initialize the feature extractor with TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to extract.
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.feature_names = None
        self.last_texts = None
        self.mean_tfidf = None
        
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from text.
        
        Args:
            texts: List of text documents to extract features from.
            
        Returns:
            Sparse matrix of TF-IDF features.
            
        Raises:
            Exception: If there's an error extracting features.
        """
        try:
            self.last_texts = texts  # Store for potential later use
            features = self.tfidf_vectorizer.fit_transform(texts)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Pre-calculate and store mean TF-IDF scores for top features extraction
            self.mean_tfidf = features.mean(axis=0).A1
            
            logger.info(f"Features extracted successfully. Shape: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def transform_features(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using fitted vectorizer.
        
        Args:
            texts: List of text documents to transform.
            
        Returns:
            Sparse matrix of TF-IDF features.
            
        Raises:
            ValueError: If vectorizer hasn't been fitted.
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer not fitted. Call extract_features first.")
        return self.tfidf_vectorizer.transform(texts)

    def get_top_features(self, n_features: int = 10) -> Dict[str, float]:
        """
        Get top N features based on mean TF-IDF scores.
        
        Args:
            n_features: Number of top features to return.
            
        Returns:
            Dictionary mapping feature names to their importance scores.
            
        Raises:
            ValueError: If feature_names is None or no texts have been processed.
            Exception: If there's an error calculating top features.
        """
        try:
            if self.feature_names is None:
                raise ValueError("Feature names not available. Call extract_features first.")
            if self.mean_tfidf is None:
                raise ValueError("TF-IDF scores not available. Call extract_features first.")
                
            # Use the pre-calculated mean TF-IDF scores
            top_indices = self.mean_tfidf.argsort()[-n_features:][::-1]
            return {
                self.feature_names[i]: self.mean_tfidf[i]
                for i in top_indices
            }
        except Exception as e:
            logger.error(f"Error getting top features: {str(e)}")
            raise

class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the model trainer with specified parameters.
        
        Args:
            params: Dictionary of parameters for the RandomForestClassifier.
                   If None, default parameters will be used.
        """
        self.model = None
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Train a Random Forest classification model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features (not used currently).
            y_val: Optional validation labels (not used currently).
            
        Raises:
            Exception: If there's an error during model training.
        """
        try:
            self.model = RandomForestClassifier(**self.params)
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features.
            y_test: True test labels.
            
        Returns:
            Dictionary containing classification report and confusion matrix.
            
        Raises:
            ValueError: If model hasn't been trained.
            Exception: If there's an error during evaluation.
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train method first.")
                
            y_pred = self.model.predict(X_test)
            # Get standard classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Convert decimal accuracy to percentage format
            if 'accuracy' in report:
                # Handle accuracy which is a single float
                report['accuracy'] = report['accuracy'] * 100.0
                
            # Handle class metrics and averages which are dictionaries
            for key in report:
                if key in ['macro avg', 'weighted avg'] or key.isdigit():
                    if isinstance(report[key], dict):
                        for metric in report[key]:
                            if isinstance(report[key][metric], float):
                                report[key][metric] = report[key][metric] * 100.0
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            logger.info("Model evaluation completed")
            return {
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

class MLPipeline:
    """Main class implementing the complete machine learning pipeline."""
    
    def __init__(self, max_features: int = 1000):
        """Initialize the ML pipeline with components.
        
        Args:
            max_features: Maximum number of features to extract using TF-IDF.
        """
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor(max_features=max_features)
        self.model_trainer = ModelTrainer()

    def run_pipeline(self, data_path: str, text_column: str, label_column: str,
                     test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data loading to model evaluation.
        
        Args:
            data_path: Path to the CSV data file.
            text_column: Name of the column containing text data.
            label_column: Name of the column containing labels.
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Dictionary containing evaluation results.
            
        Raises:
            Exception: If there's an error in any step of the pipeline.
        """
        try:
            # Load and preprocess data
            logger.info(f"Starting ML pipeline with data from {data_path}")
            df = self.preprocessor.load_data(data_path)
            df = self.preprocessor.preprocess_text(df, text_column)
            df, labels = self.preprocessor.encode_labels(df, label_column)

            # Extract features
            features = self.feature_extractor.extract_features(df[text_column])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state
            )
            logger.info(f"Data split: training samples={X_train.shape[0]}, test samples={X_test.shape[0]}")

            # Train model
            self.model_trainer.train(X_train, y_train)

            # Evaluate model
            evaluation_results = self.model_trainer.evaluate(X_test, y_test)

            logger.info("Pipeline completed successfully")
            return evaluation_results

        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise

def plot_confusion_matrix(conf_matrix: np.ndarray, classes: List[str], save_path: str = None) -> None:
    """
    Plot a confusion matrix as a heatmap.
    
    Args:
        conf_matrix: Confusion matrix to plot.
        classes: Class labels for axis labels.
        save_path: Optional path to save the figure instead of displaying it.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def plot_top_features(feature_dict: Dict[str, float], save_path: str = None) -> None:
    """
    Plot top features by their TF-IDF scores.
    
    Args:
        feature_dict: Dictionary mapping feature names to their importance scores.
        save_path: Optional path to save the figure instead of displaying it.
    """
    plt.figure(figsize=(12, 6))
    features = list(feature_dict.keys())
    scores = list(feature_dict.values())
    
    plt.barh(features, scores)
    plt.title('Top Features by TF-IDF Score')
    plt.xlabel('Mean TF-IDF Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Top features plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    pipeline = MLPipeline(max_features=1000)
    results = pipeline.run_pipeline(
        data_path="your_data.csv",
        text_column="text_column",
        label_column="label_column"
    )
    
    # Print results
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).transpose())
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        classes=pipeline.preprocessor.label_encoder.classes_
    )
    
    # Plot top features if available
    try:
        top_features = pipeline.feature_extractor.get_top_features(n_features=10)
        plot_top_features(top_features)
    except Exception as e:
        logger.warning(f"Could not plot top features: {str(e)}")