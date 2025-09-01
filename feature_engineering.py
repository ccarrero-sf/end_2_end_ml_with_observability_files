"""
Feature engineering module for the End-to-End ML Workflow with Observability.

This module handles the creation of customer behavioral features for churn prediction,
including time-series aggregations, sentiment analysis features, and customer profiling.
"""

import json
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.ml.feature_store import Entity, FeatureView

from config import config, feature_config, logger


class FeatureEngineer:
    """Main class for feature engineering operations."""
    
    def __init__(self, session: Session, db: str, sc: str):
        """
        Initialize the feature engineer.
        
        Args:
            session: Snowflake session
            db: Database name  
            sc: Schema name
        """
        self.session = session
        self.db = db
        self.sc = sc
    
    def create_customer_features(self, cur_date: datetime, table_name: str):
        """
        Create comprehensive customer behavioral features for churn prediction.
        
        Args:
            cur_date: Current date for feature calculation
            table_name: Target table name for features
        """
        logger.info(f"Creating customer features for {cur_date}")
        
        # Load data tables
        customers_df = self.session.table(f'{self.db}.{self.sc}.CUSTOMERS')
        sales_df = self.session.table(f'{self.db}.{self.sc}.SALES').filter(F.col("TRANSACTION_DATE") < F.lit(cur_date))
        feedback_df = self.session.table(f'{self.db}.{self.sc}.FEEDBACK_SENTIMENT').filter(F.col("CHAT_DATE") < F.lit(cur_date))
        
        # Validate minimum data requirements
        if customers_df.count() < 10:
            raise ValueError("Insufficient customer data")
        print (f'sales data count: {sales_df.count()}')

        if sales_df.count() < 10:
            raise ValueError("Insufficient sales data")
        
        # Create sales aggregations by customer
        sales_agg_df = sales_df.group_by("CUSTOMER_ID").agg(
            F.max("TRANSACTION_DATE").alias("LAST_PURCHASE_DATE"),
            F.sum("TOTAL_AMOUNT").alias("TOTAL_CUSTOMER_VALUE")
        )
        
        # Create time-series aggregations for sales
        sales_ts_df = sales_df.analytics.time_series_agg(
            time_col="TRANSACTION_DATE",
            aggs={"TOTAL_AMOUNT": ["SUM", "COUNT"]},
            windows=["-7D", "-1MM", "-2MM", "-3MM"],
            sliding_interval="1D",
            group_by=["CUSTOMER_ID"],
            col_formatter=lambda input_col, agg, window: f"{agg}_{input_col}_{window.replace('-', 'PAST_')}"
        )
        
        # Join sales features
        sales_features_df = sales_agg_df.join(
            sales_ts_df,
            (sales_agg_df.LAST_PURCHASE_DATE == sales_ts_df.TRANSACTION_DATE) &
            (sales_agg_df.CUSTOMER_ID == sales_ts_df.CUSTOMER_ID),
            "left"
        ).select(
            sales_agg_df["CUSTOMER_ID"].alias("CUSTOMER_ID"),
            sales_agg_df["TOTAL_CUSTOMER_VALUE"],
            sales_agg_df["LAST_PURCHASE_DATE"],
            sales_ts_df["SUM_TOTAL_AMOUNT_PAST_7D"],
            sales_ts_df["SUM_TOTAL_AMOUNT_PAST_1MM"],
            sales_ts_df["SUM_TOTAL_AMOUNT_PAST_2MM"],
            sales_ts_df["SUM_TOTAL_AMOUNT_PAST_3MM"],
            sales_ts_df["COUNT_TOTAL_AMOUNT_PAST_7D"].alias("COUNT_ORDERS_PAST_7D"),
            sales_ts_df["COUNT_TOTAL_AMOUNT_PAST_1MM"].alias("COUNT_ORDERS_PAST_1MM"),
            sales_ts_df["COUNT_TOTAL_AMOUNT_PAST_2MM"].alias("COUNT_ORDERS_PAST_2MM"),
            sales_ts_df["COUNT_TOTAL_AMOUNT_PAST_3MM"].alias("COUNT_ORDERS_PAST_3MM")
        )
        
        # Create feedback features if data exists
        if feedback_df.count() > 0:
            # Get latest feedback date for each customer
            latest_feedback_df = feedback_df.group_by("CUSTOMER_ID").agg(
                F.max("CHAT_DATE").alias("CHAT_DATE")
            )
            
            # Create moving aggregations for sentiment
            feedback_agg_df = feedback_df.analytics.moving_agg(
                aggs={"SENTIMENT": ["MIN", "AVG"]},
                window_sizes=[2, 3, 4],
                order_by=["CHAT_DATE"],
                group_by=["CUSTOMER_ID"]
            )
            
            # Join feedback features
            feedback_features_df = latest_feedback_df.join(
                feedback_agg_df, "CUSTOMER_ID", "left"
            ).select(
                latest_feedback_df["CUSTOMER_ID"],
                feedback_agg_df["SENTIMENT_MIN_2"],
                feedback_agg_df["SENTIMENT_MIN_3"],
                feedback_agg_df["SENTIMENT_MIN_4"],
                feedback_agg_df["SENTIMENT_AVG_2"],
                feedback_agg_df["SENTIMENT_AVG_3"],
                feedback_agg_df["SENTIMENT_AVG_4"]
            )
        else:
            # Create empty feedback features
            logger.warning("No feedback data available, setting sentiment features to null")
            feedback_features_df = None
        
        # Combine all features
        features_df = customers_df.join(sales_features_df, "CUSTOMER_ID", "left")
        
        if feedback_features_df is not None:
            features_df = features_df.join(feedback_features_df, "CUSTOMER_ID", "left")
        else:
            # Add null feedback columns when no feedback data
            for col in ["SENTIMENT_MIN_2", "SENTIMENT_MIN_3", "SENTIMENT_MIN_4", 
                       "SENTIMENT_AVG_2", "SENTIMENT_AVG_3", "SENTIMENT_AVG_4"]:
                features_df = features_df.with_column(col, F.lit(None).cast("float"))
        
        # Select final columns and add derived features
        features_df = features_df.select(
            customers_df["CUSTOMER_ID"],
            customers_df["AGE"],
            customers_df["GENDER"],
            customers_df["LOCATION"],
            customers_df["CUSTOMER_SEGMENT"],
            sales_features_df["LAST_PURCHASE_DATE"],
            F.col("SENTIMENT_MIN_2"),
            F.col("SENTIMENT_MIN_3"),
            F.col("SENTIMENT_MIN_4"),
            F.col("SENTIMENT_AVG_2"),
            F.col("SENTIMENT_AVG_3"),
            F.col("SENTIMENT_AVG_4"),
            sales_features_df["SUM_TOTAL_AMOUNT_PAST_7D"],
            sales_features_df["SUM_TOTAL_AMOUNT_PAST_1MM"],
            sales_features_df["SUM_TOTAL_AMOUNT_PAST_2MM"],
            sales_features_df["SUM_TOTAL_AMOUNT_PAST_3MM"],
            sales_features_df["COUNT_ORDERS_PAST_7D"],
            sales_features_df["COUNT_ORDERS_PAST_1MM"],
            sales_features_df["COUNT_ORDERS_PAST_2MM"],
            sales_features_df["COUNT_ORDERS_PAST_3MM"],
            F.datediff("day", sales_features_df["LAST_PURCHASE_DATE"], F.lit(cur_date)).alias("DAYS_SINCE_LAST_PURCHASE"),
            F.lit(cur_date).alias("TIMESTAMP")
        ).filter(
            sales_features_df["LAST_PURCHASE_DATE"].isNotNull()
        ).dropDuplicates(["CUSTOMER_ID", "TIMESTAMP"])
        
        # Fill null values with 0 for numerical features
        fill_columns = [
            "SENTIMENT_MIN_2", "SENTIMENT_MIN_3", "SENTIMENT_MIN_4", 
            "SENTIMENT_AVG_2", "SENTIMENT_AVG_3", "SENTIMENT_AVG_4",
            "SUM_TOTAL_AMOUNT_PAST_7D", "SUM_TOTAL_AMOUNT_PAST_1MM", 
            "SUM_TOTAL_AMOUNT_PAST_2MM", "SUM_TOTAL_AMOUNT_PAST_3MM",
            "COUNT_ORDERS_PAST_7D", "COUNT_ORDERS_PAST_1MM", 
            "COUNT_ORDERS_PAST_2MM", "COUNT_ORDERS_PAST_3MM"
        ]
        
        for column in fill_columns:
            features_df = features_df.fillna({column: 0})
        
        # Save features
        features_df.write.mode("append").save_as_table(table_name)
        
        logger.info(f"Features created for {cur_date} - {features_df.count()} records")


class FeatureStoreManager:
    """Manager for Feature Store operations."""
    
    def __init__(self, session: Session, fs, db: str, sc: str):
        """
        Initialize Feature Store manager.
        
        Args:
            session: Snowflake session
            fs: Feature Store instance
            db: Database name
            sc: Schema name
        """
        self.session = session
        self.fs = fs
        self.db = db
        self.sc = sc
    
    def setup_feature_store_complete(self, labeled_df):
        """
        Setup complete Feature Store with entities and feature views.
        
        Args:
            labeled_df: Labeled features dataframe
            
        Returns:
            Feature view instance
        """
        logger.info("Setting up Feature Store...")
        
        # Create customer entity if it doesn't exist
        entity_name = "CUSTOMER_ENT"
        try:
            existing_entities = json.loads(
                self.fs.list_entities().select(F.to_json(F.array_agg("NAME", True))).collect()[0][0]
            )
            
            if entity_name not in existing_entities:
                customer_entity = Entity(
                    name=entity_name, 
                    join_keys=["CUSTOMER_ID"],
                    desc="Primary Key for CUSTOMER"
                )
                self.fs.register_entity(customer_entity)
                logger.info(f"Entity {entity_name} created")
            else:
                customer_entity = self.fs.get_entity(entity_name)
                logger.info(f"Entity {entity_name} already exists")
            
            # Ensure we're in the correct schema after Feature Store entity operations
            self.session.sql(f'USE SCHEMA {self.sc}').collect()
                
        except Exception as e:
            logger.error(f"Error creating entity {entity_name}: {str(e)}")
            raise
        
        # Create feature view if it doesn't exist
        fv_name = "FV_CUSTOMER_CHURN"
        fv_version = "V_1"
        
        try:
            feature_view = self.fs.get_feature_view(name=fv_name, version=fv_version)
            logger.info(f"Feature View {fv_name}_{fv_version} already exists")
        except:
            feature_view_instance = FeatureView(
                name=fv_name, 
                entities=[customer_entity], 
                feature_df=labeled_df,
                timestamp_col="TIMESTAMP",
                refresh_freq=None,
                desc="Features for customer churn detection"
            )
            
            feature_view = self.fs.register_feature_view(
                feature_view=feature_view_instance, 
                version=fv_version, 
                block=True
            )
            logger.info(f"Feature View {fv_name}_{fv_version} created")
        
        # Ensure we're in the correct schema after Feature Store operations
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        logger.info("Feature Store setup completed")
        return feature_view
    
    def create_dataset_from_feature_store(self, feature_view, name: str, timestamp, version: str = 'v1'):
        """
        Create dataset from Feature Store with proper data type handling.
        
        Args:
            feature_view: Feature view instance
            name: Dataset name
            timestamp: Timestamp for dataset
            version: Dataset version
            
        Returns:
            Snowpark DataFrame with features
        """
        # Create spine dataframe
        spine_df = feature_view.feature_df.filter(
            F.col("TIMESTAMP") == F.lit(timestamp)
        ).group_by('CUSTOMER_ID').agg(F.max('TIMESTAMP').as_('TIMESTAMP'))
        
        # Generate dataset
        dataset = self.fs.generate_dataset(
            name=name, 
            version=version,
            spine_df=spine_df, 
            features=[feature_view], 
            spine_timestamp_col='TIMESTAMP'
        )
        
        # Ensure we're in the correct schema after Feature Store dataset generation
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        # Convert to Snowpark DataFrame and handle data types
        dataset_df = dataset.read.to_snowpark_dataframe()
        
        # Convert decimal columns to float for ML compatibility
        decimal_columns = [field.name for field in dataset_df.schema.fields
                          if isinstance(field.datatype, config.NUMERIC_TYPES)]
        
        for column_name in decimal_columns:
            dataset_df = dataset_df.with_column(
                column_name,
                F.col(column_name).cast("float")
            )
        
        logger.info(f"Dataset {name} created with {dataset_df.count()} records")
        return dataset_df
