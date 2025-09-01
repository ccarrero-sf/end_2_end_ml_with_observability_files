"""
Model monitoring module for the End-to-End ML Workflow with Observability.

This module handles model inference, monitoring, and performance tracking
for the customer churn prediction pipeline.
"""

from sklearn.metrics import f1_score
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from config import config, logger


class ModelMonitor:
    """Main class for model monitoring operations."""
    
    def __init__(self, session: Session, mr, mr_schema: str, sc: str):
        """
        Initialize the model monitor.
        
        Args:
            session: Snowflake session
            mr: Model Registry instance
            mr_schema: Model Registry schema name
            sc: Current schema name
        """
        self.session = session
        self.mr = mr
        self.mr_schema = mr_schema
        self.sc = sc
    
    def setup_complete_monitoring_infrastructure(self):
        """Setup complete model monitoring infrastructure including prediction tables and monitors."""
        logger.info("Setting up complete model monitoring infrastructure...")
        
        # Create prediction tables for monitoring
        monitoring_tables = ["customer_churn_baseline_predicted", "CUSTOMER_CHURN_PREDICTED_PROD2"]
        
        for table in monitoring_tables:
            self.session.sql(f"""       
                CREATE OR REPLACE TABLE {table} (
                    CUSTOMER_ID VARCHAR(16777216),
                    TIMESTAMP TIMESTAMP_NTZ(9),
                    GENDER VARCHAR(16777216),
                    LOCATION VARCHAR(16777216),
                    CUSTOMER_SEGMENT VARCHAR(16777216),
                    LAST_PURCHASE_DATE DATE,
                    NEXT_TRANSACTION_DATE DATE,
                    AGE FLOAT,
                    SENTIMENT_MIN_2 FLOAT,
                    SENTIMENT_MIN_3 FLOAT,
                    SENTIMENT_MIN_4 FLOAT,
                    SENTIMENT_AVG_2 FLOAT,
                    SENTIMENT_AVG_3 FLOAT,
                    SENTIMENT_AVG_4 FLOAT,
                    SUM_TOTAL_AMOUNT_PAST_7D FLOAT,
                    SUM_TOTAL_AMOUNT_PAST_1MM FLOAT,
                    SUM_TOTAL_AMOUNT_PAST_2MM FLOAT,
                    SUM_TOTAL_AMOUNT_PAST_3MM FLOAT,
                    COUNT_ORDERS_PAST_7D FLOAT,
                    COUNT_ORDERS_PAST_1MM FLOAT,
                    COUNT_ORDERS_PAST_2MM FLOAT,
                    COUNT_ORDERS_PAST_3MM FLOAT,
                    DAYS_SINCE_LAST_PURCHASE FLOAT,
                    CHURNED FLOAT,
                    CHURNED_PRED_PROD FLOAT,
                    CHURNED_PRED_BASE FLOAT,
                    CHURNED_PRED_RETRAIN FLOAT,
                    CHURNED_PRED_PROBABILITY FLOAT,
                    VERSION_NAME VARCHAR(50)
                )
            """).collect()
        
        # Create Model Monitors
        self.session.sql(f'USE SCHEMA {self.mr_schema}').collect()
        
        try:
            # Create fake prod and retrain model versions for monitoring
            # (Each monitor requires a unique model version)
            baseline_model = self.mr.get_model(config.MODEL_NAME).version("baseline")
            
            fake_prod_model = self.mr.log_model(
                model=baseline_model,
                model_name=config.MODEL_NAME,
                version_name="PRODMONITOR"
            )
            
            fake_retrain_model = self.mr.log_model(
                model=baseline_model,
                model_name=config.MODEL_NAME,
                version_name="RETRAIN"
            )
            
            # Ensure we're in the correct schema after Model Registry operations
            self.session.sql(f'USE SCHEMA {self.mr_schema}').collect()
            
            # Monitor for baseline model
            self.session.sql(f"""
                CREATE OR REPLACE MODEL MONITOR Monitor_ChurnDetector_Base
                WITH
                    MODEL={config.MODEL_NAME}
                    VERSION=baseline
                    FUNCTION=predict
                    SOURCE={self.sc}.CUSTOMER_CHURN_PREDICTED_PROD2
                    BASELINE={self.sc}.customer_churn_baseline_predicted
                    TIMESTAMP_COLUMN=TIMESTAMP
                    PREDICTION_CLASS_COLUMNS=(CHURNED_PRED_BASE)  
                    ACTUAL_CLASS_COLUMNS=(CHURNED)
                    ID_COLUMNS=(CUSTOMER_ID)
                    WAREHOUSE={config.WAREHOUSE}
                    REFRESH_INTERVAL='{config.MONITOR_REFRESH_INTERVAL}'
                    AGGREGATION_WINDOW='{config.MONITOR_AGGREGATION_WINDOW}';
            """).collect()
            
            # Monitor for production model
            self.session.sql(f"""
                CREATE OR REPLACE MODEL MONITOR Monitor_ChurnDetector_Prod
                WITH
                    MODEL={config.MODEL_NAME}
                    VERSION=PRODMONITOR
                    FUNCTION=predict
                    SOURCE={self.sc}.CUSTOMER_CHURN_PREDICTED_PROD2
                    BASELINE={self.sc}.customer_churn_baseline_predicted
                    TIMESTAMP_COLUMN=TIMESTAMP
                    PREDICTION_CLASS_COLUMNS=(CHURNED_PRED_PROD)  
                    ACTUAL_CLASS_COLUMNS=(CHURNED)
                    ID_COLUMNS=(CUSTOMER_ID)
                    WAREHOUSE={config.WAREHOUSE}
                    REFRESH_INTERVAL='{config.MONITOR_REFRESH_INTERVAL}'
                    AGGREGATION_WINDOW='{config.MONITOR_AGGREGATION_WINDOW}';
            """).collect()
            
            # Monitor for retrained model
            self.session.sql(f"""
                CREATE OR REPLACE MODEL MONITOR Monitor_ChurnDetector_Retrain
                WITH
                    MODEL={config.MODEL_NAME}
                    VERSION=RETRAIN
                    FUNCTION=predict
                    SOURCE={self.sc}.CUSTOMER_CHURN_PREDICTED_PROD2
                    BASELINE={self.sc}.customer_churn_baseline_predicted
                    TIMESTAMP_COLUMN=TIMESTAMP
                    PREDICTION_CLASS_COLUMNS=(CHURNED_PRED_RETRAIN)  
                    ACTUAL_CLASS_COLUMNS=(CHURNED)
                    ID_COLUMNS=(CUSTOMER_ID)
                    WAREHOUSE={config.WAREHOUSE}
                    REFRESH_INTERVAL='{config.MONITOR_REFRESH_INTERVAL}'
                    AGGREGATION_WINDOW='{config.MONITOR_AGGREGATION_WINDOW}';
            """).collect()
            
            logger.info("Model Monitors created successfully:")
            logger.info("  - Monitor_ChurnDetector_Base")
            logger.info("  - Monitor_ChurnDetector_Prod")
            logger.info("  - Monitor_ChurnDetector_Retrain")
            
        finally:
            # Switch back to original schema
            self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        logger.info("Monitoring infrastructure setup completed")
    
    def run_inference_and_store(self, model, dataset_df, output_table: str, col_name: str):
        """
        Run inference and store results for monitoring.
        
        Args:
            model: Model instance for inference
            dataset_df: Input dataset
            output_table: Output table for predictions
            col_name: Column name for predictions
        """
        logger.info(f"Running inference for {col_name}")
        
        # Get predictions
        predictions = model.run(dataset_df, function_name="predict")
        predictions = predictions.select([F.col(c).alias(c.replace('"', '')) for c in predictions.columns])
        predictions_df = predictions.rename("output_feature_0", col_name)
        predictions_df = predictions_df.with_column("VERSION_NAME", F.lit(model.version_name))
        predictions_df = predictions_df.with_column("CHURNED_PRED_PROBABILITY", F.col(col_name))
        
        # Store in temporary table first
        predictions_df.write.mode("overwrite").save_as_table('TEMP_PREDICTIONS')
        
        # Merge with output table
        output_columns = [field.name for field in self.session.table(output_table).schema]
        insert_columns = ", ".join(output_columns)
        insert_values = ", ".join([
            f"t.{col}" if col in predictions_df.columns else "NULL" for col in output_columns
        ])
        
        merge_statement = f"""
            MERGE INTO {output_table} o
            USING TEMP_PREDICTIONS t
            ON o.CUSTOMER_ID = t.CUSTOMER_ID AND o.TIMESTAMP = t.TIMESTAMP
            WHEN MATCHED THEN
                UPDATE SET o.{col_name} = t.{col_name},
                           o.VERSION_NAME = t.VERSION_NAME,
                           o.CHURNED_PRED_PROBABILITY = t.CHURNED_PRED_PROBABILITY
            WHEN NOT MATCHED THEN
                INSERT ({insert_columns})
                VALUES ({insert_values})
        """
        
        self.session.sql(merge_statement).collect()
        logger.info(f"Predictions stored in {output_table}")
    
    def get_model_performance_sklearn(self, prediction_column: str, table_name: str = 'CUSTOMER_CHURN_PREDICTED_PROD2', 
                                    limit_latest_timestamp: bool = True) -> float:
        """
        Get F1 score using sklearn directly from the prediction table.
        
        Args:
            prediction_column: Column containing predictions
            table_name: Table containing predictions and true labels
            limit_latest_timestamp: Whether to filter for latest timestamp only
            
        Returns:
            F1 score calculated using sklearn
        """
        # Validate prediction column
        valid_columns = ['CHURNED_PRED_PROD', 'CHURNED_PRED_BASE', 'CHURNED_PRED_RETRAIN']
        if prediction_column not in valid_columns:
            raise ValueError(f"prediction_column must be one of {valid_columns}")
        
        try:
            # Get the data from the table
            table_df = self.session.table(table_name)
            
            if limit_latest_timestamp:
                # Get latest timestamp with labels
                labeled_table = config.TABLES['labeled']
                timestamps = self.session.table(labeled_table).select("TIMESTAMP").distinct().sort("TIMESTAMP").collect()
                if len(timestamps) >= 2:
                    latest_timestamp = timestamps[-2]["TIMESTAMP"]
                    table_df = table_df.filter(F.col("TIMESTAMP") == latest_timestamp)
            
            # Get only rows where both true labels and predictions are not null
            filtered_df = table_df.filter(
                (F.col("CHURNED").is_not_null()) & 
                (F.col(prediction_column).is_not_null())
            ).select("CHURNED", prediction_column)
            
            # Convert to pandas for sklearn
            pandas_df = filtered_df.to_pandas()
            
            if len(pandas_df) == 0:
                logger.warning(f"No data available for {prediction_column}")
                return 1.0  # Default high score if no data
            
            # Calculate F1 score using sklearn
            y_true = pandas_df['CHURNED'].astype(int)
            y_pred = pandas_df[prediction_column].astype(int)
            
            f1 = f1_score(y_true, y_pred)
            
            return f1
            
        except Exception as e:
            logger.warning(f"Error calculating F1 score for {prediction_column}: {str(e)}")
            return 1.0  # Default if calculation fails
