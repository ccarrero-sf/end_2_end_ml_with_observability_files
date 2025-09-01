"""
Configuration module for the End-to-End ML Workflow with Observability.

This module contains all configuration variables and constants used throughout
the ML pipeline for customer churn detection.
"""

import os
from datetime import datetime
from snowflake.snowpark.types import DecimalType, FloatType, IntegerType, DoubleType, LongType

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

class Config:
    """Main configuration class for the ML pipeline."""
    
    # Snowflake Environment Settings
    # These can be overridden with environment variables ML_DATABASE, ML_SCHEMA, ML_WAREHOUSE
    DATABASE = os.getenv('ML_DATABASE', 'CC_SANDBOX_ML2')  # Main database for the ML pipeline
    SCHEMA = os.getenv('ML_SCHEMA', 'E2E_DEMO')     # Main schema for the ML pipeline
    WAREHOUSE = os.getenv('ML_WAREHOUSE', 'COMPUTE_WH')
    
    # ML Parameters
    CHURN_WINDOW = int(os.getenv('CHURN_WINDOW', 30))  # Days to define churn
    
    # Model Training Parameters
    RETRAIN_THRESHOLD = float(os.getenv('RETRAIN_THRESHOLD', 0.8))
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    
    # Data Processing
    STAGE_NAME = "CSV"
    S3_URL = 's3://sfquickstarts/vhol_end_2_end_ml_with_observability/'
    
    # Feature Engineering
    CATEGORICAL_COLS = ['GENDER', 'LOCATION', 'CUSTOMER_SEGMENT']
    NUMERICAL_COLS = [
        "AGE", "SENTIMENT_MIN_2", "SENTIMENT_MIN_3", "SENTIMENT_MIN_4", 
        "SENTIMENT_AVG_2", "SENTIMENT_AVG_3", "SENTIMENT_AVG_4",
        "SUM_TOTAL_AMOUNT_PAST_7D", "SUM_TOTAL_AMOUNT_PAST_1MM", 
        "SUM_TOTAL_AMOUNT_PAST_2MM", "SUM_TOTAL_AMOUNT_PAST_3MM",
        "COUNT_ORDERS_PAST_7D", "COUNT_ORDERS_PAST_1MM", 
        "COUNT_ORDERS_PAST_2MM", "COUNT_ORDERS_PAST_3MM"
    ]
    
    @property
    def feature_cols(self):
        """Combined feature columns for model training."""
        return self.CATEGORICAL_COLS + self.NUMERICAL_COLS
    
    TARGET_COL = "CHURNED"
    
    # Table Names
    TABLES = {
        'sales': 'SALES',
        'customers': 'CUSTOMERS',
        'feedback_raw': 'FEEDBACK_RAW',
        'feedback_sentiment': 'FEEDBACK_SENTIMENT',
        'features': 'CUSTOMER_FEATURES',
        'labeled': 'CUSTOMER_FEATURES_LABELED',
        'baseline_predictions': 'customer_churn_baseline_predicted',
        'prod_predictions': 'CUSTOMER_CHURN_PREDICTED_PROD2',
        'files_ingested': 'FILES_INGESTED'
    }
    
    # Model Registry
    MODEL_NAME = "ChurnDetector"
    MODEL_DEPENDENCIES = ["snowflake-ml-python", "xgboost", "scikit-learn"]
    
    # Data Types
    NUMERIC_TYPES = (DecimalType, FloatType, IntegerType, DoubleType, LongType)
    
    # Monitoring
    MONITOR_REFRESH_INTERVAL = '1 min'
    MONITOR_AGGREGATION_WINDOW = '1 day'
    
    def __init__(self):
        """Initialize configuration and validate required settings."""
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration parameters."""
        if self.CHURN_WINDOW <= 0:
            raise ValueError("CHURN_WINDOW must be positive")
        
        if not 0 < self.RETRAIN_THRESHOLD <= 1:
            raise ValueError("RETRAIN_THRESHOLD must be between 0 and 1")
        
        if not 0 < self.TEST_SIZE < 1:
            raise ValueError("TEST_SIZE must be between 0 and 1")
        
        if not self.DATABASE:
            raise ValueError("DATABASE must be specified")
        
        if not self.SCHEMA:
            raise ValueError("SCHEMA must be specified")
    
    def get_schema_names(self, db: str, sc: str):
        """Get derived schema names for Feature Store and Model Registry."""
        return {
            'model_registry': f'{sc}_MODEL_REGISTRY'.replace('"', ''),
            'feature_store': f'{sc}_FEATURE_STORE'.replace('"', '')
        }
    
    def get_table_name(self, table_key: str, db: str = None, sc: str = None):
        """Get fully qualified table name."""
        table_name = self.TABLES.get(table_key, table_key)
        if db and sc:
            return f'{db}.{sc}.{table_name}'
        return table_name

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

def setup_logging(level=logging.INFO):
    """Setup logging configuration for the ML pipeline."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_pipeline.log')
        ]
    )
    return logging.getLogger("e2e-ml-workflow")

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

class FeatureConfig:
    """Configuration specific to feature engineering."""
    
    # Time series aggregation windows
    TIME_WINDOWS = ["-7D", "-1MM", "-2MM", "-3MM"]
    
    # Moving average windows for sentiment
    SENTIMENT_WINDOWS = [2, 3, 4]
    
    # Aggregation functions for sales data
    SALES_AGGS = {"TOTAL_AMOUNT": ["SUM", "COUNT"]}
    
    # Aggregation functions for sentiment data
    SENTIMENT_AGGS = {"SENTIMENT": ["MIN", "AVG"]}
    
    # Columns to fill with zero for null values
    FILL_ZERO_COLUMNS = [
        "SENTIMENT_MIN_2", "SENTIMENT_MIN_3", "SENTIMENT_MIN_4", 
        "SENTIMENT_AVG_2", "SENTIMENT_AVG_3", "SENTIMENT_AVG_4",
        "SUM_TOTAL_AMOUNT_PAST_7D", "SUM_TOTAL_AMOUNT_PAST_1MM", 
        "SUM_TOTAL_AMOUNT_PAST_2MM", "SUM_TOTAL_AMOUNT_PAST_3MM",
        "COUNT_ORDERS_PAST_7D", "COUNT_ORDERS_PAST_1MM", 
        "COUNT_ORDERS_PAST_2MM", "COUNT_ORDERS_PAST_3MM"
    ]
    
    @staticmethod
    def custom_column_naming(input_col, agg, window):
        """Custom column naming for time-series features."""
        return f"{agg}_{input_col}_{window.replace('-', 'PAST_')}"

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create global configuration instance
config = Config()
feature_config = FeatureConfig()
logger = setup_logging()
