"""
Model training module for the End-to-End ML Workflow with Observability.

This module handles ML model training, evaluation, and registration
for the customer churn prediction pipeline.
"""

from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from snowflake.snowpark import Session
from snowflake.ml.model import type_hints

from config import config, logger


class ModelTrainer:
    """Main class for model training operations."""
    
    def __init__(self, session: Session, mr, db: str, sc: str):
        """
        Initialize the model trainer.
        
        Args:
            session: Snowflake session
            mr: Model Registry instance
            db: Database name
            sc: Schema name
        """
        self.session = session
        self.mr = mr
        self.db = db
        self.sc = sc
    
    def train_and_register_model(self, feature_df, version_name: str, sample_data, 
                                comment: Optional[str] = None, set_as_default: bool = False):
        """
        Train XGBoost model, evaluate it, and register in Model Registry.
        
        Args:
            feature_df: Snowpark DataFrame with features
            version_name: Version name for the model
            sample_data: Sample input data for model signature
            comment: Optional comment for the model version
            set_as_default: Whether to set as default version
            
        Returns:
            Dictionary with model instance and metrics
        """
        logger.info("Training churn prediction model...")
        
        # Convert to pandas for sklearn
        train_df = feature_df.to_pandas()
        
        # Validate training data
        if len(train_df) < 100:
            raise ValueError(f"Insufficient training data: {len(train_df)} samples")
        
        # Check for missing required columns
        missing_cols = [col for col in config.feature_cols + [config.TARGET_COL] 
                       if col not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check label distribution
        positive_rate = train_df[config.TARGET_COL].mean()
        if positive_rate < 0.01 or positive_rate > 0.99:
            logger.warning(f"Extreme label distribution: {positive_rate:.2%} positive samples")
        
        logger.info(f"Training data validation passed: {len(train_df)} samples, {positive_rate:.2%} positive")
        
        # Split data
        train_data, test_data = train_test_split(
            train_df, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=train_df[config.TARGET_COL]
        )
        
        logger.info(f"Training set: {len(train_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
                 config.CATEGORICAL_COLS),
                ("scaler", StandardScaler(), config.NUMERICAL_COLS)
            ]
        )
        
        # Create model pipeline
        pipeline = Pipeline(
            steps=[ 
                ("preprocessor", preprocessor),
                ("model", XGBClassifier(
                    random_state=config.RANDOM_STATE,
                    eval_metric='logloss'
                ))
            ]
        )
        
        # Train model
        X_train = train_data[config.feature_cols]
        y_train = train_data[config.TARGET_COL]
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate model on train and test sets
        metrics = {}
        
        # Training metrics
        train_predictions = pipeline.predict(X_train)
        metrics['train_f1_score'] = f1_score(y_train, train_predictions)
        metrics['train_precision'] = precision_score(y_train, train_predictions)
        metrics['train_recall'] = recall_score(y_train, train_predictions)
        metrics['train_accuracy'] = accuracy_score(y_train, train_predictions)
        
        # Test metrics
        X_test = test_data[config.feature_cols]
        y_test = test_data[config.TARGET_COL]
        
        test_predictions = pipeline.predict(X_test)
        metrics['test_f1_score'] = f1_score(y_test, test_predictions)
        metrics['test_precision'] = precision_score(y_test, test_predictions)
        metrics['test_recall'] = recall_score(y_test, test_predictions)
        metrics['test_accuracy'] = accuracy_score(y_test, test_predictions)
        
        logger.info("Model training completed:")
        logger.info(f"  Training F1 Score: {metrics['train_f1_score']:.4f}")
        logger.info(f"  Test F1 Score: {metrics['test_f1_score']:.4f}")
        
        # Register model in Model Registry
        logger.info(f"Registering model version: {version_name}")
        
        model_instance = self.mr.log_model(
            model=pipeline,
            model_name=config.MODEL_NAME,
            version_name=version_name,
            conda_dependencies=config.MODEL_DEPENDENCIES,
            sample_input_data=sample_data,
            task=type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
            target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],                
            comment=comment or f"Churn detection model - {version_name}"
        )
        
        # Ensure we're in the correct schema after Model Registry operation
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        # Set metrics
        for metric_name, metric_value in metrics.items():
            model_instance.set_metric(metric_name=metric_name, value=metric_value)
        
        # Set as default if requested
        if set_as_default:
            mr_schema = f'{self.sc}_MODEL_REGISTRY'.replace('"', '')
            self.session.sql(f'USE SCHEMA {mr_schema}').collect()
            self.session.sql(f'ALTER MODEL {config.MODEL_NAME} SET DEFAULT_VERSION = {version_name};').collect()
            self.session.sql(f'USE SCHEMA {self.sc}').collect()
            logger.info(f"Default model set to: {version_name}")
        
        logger.info(f"Model {version_name} registered successfully")
        
        return {
            'model_instance': model_instance,
            'metrics': metrics,
            'pipeline': pipeline
        }
    
    def evaluate_model_on_validation(self, model, validation_dataset) -> Dict:
        """
        Evaluate model on validation dataset.
        
        Args:
            model: Registered model instance
            validation_dataset: Validation dataset
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Evaluating model on validation dataset...")
        
        validation_df = validation_dataset.to_pandas()
        
        # Get predictions
        val_predictions = model.run(validation_dataset, function_name="predict")
        val_predictions_df = val_predictions.to_pandas()
        
        # Calculate metrics
        y_true = validation_df[config.TARGET_COL]
        y_pred = val_predictions_df['output_feature_0']  # Default output column name
        
        validation_metrics = {
            'validation_f1_score': f1_score(y_true, y_pred),
            'validation_precision': precision_score(y_true, y_pred),
            'validation_recall': recall_score(y_true, y_pred),
            'validation_accuracy': accuracy_score(y_true, y_pred)
        }
        
        logger.info(f"Validation F1 Score: {validation_metrics['validation_f1_score']:.4f}")
        
        return validation_metrics
