"""
Utility functions for the End-to-End ML Workflow with Observability.

This module contains common utility functions used across different
components of the ML pipeline.
"""

import re
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from config import config, logger


class DatabaseUtils:
    """Utility functions for database operations."""
    
    @staticmethod
    def setup_environment(session: Session, db: str, sc: str) -> Dict[str, str]:
        """
        Setup the ML environment including schemas and staging areas.
        
        Args:
            session: Snowflake session
            db: Database name
            sc: Schema name
            
        Returns:
            Dictionary with schema names
        """
        # Create and use dedicated schema
        session.sql(f'CREATE OR REPLACE SCHEMA {sc}').collect()
        session.sql(f'USE SCHEMA {sc}').collect()
        session.sql('CREATE OR REPLACE STAGE ML_STAGE').collect()
        
        logger.info(f"Environment configured - Schema: {sc}")
        logger.info(f"Working in Database: {db}, Schema: {sc}")
        
        # Get derived schema names
        schema_names = config.get_schema_names(db, sc)
        
        return schema_names
    
    @staticmethod
    def create_core_tables(session: Session):
        """Create core data tables for the ML pipeline."""
        logger.info("Creating core data tables...")
        
        # Create SALES table
        session.sql("""
        CREATE OR REPLACE TABLE SALES (
            TRANSACTION_ID VARCHAR,
            CUSTOMER_ID VARCHAR,
            TRANSACTION_DATE DATE,
            DISCOUNT_APPLIED BOOLEAN,
            NUM_ITEMS NUMBER,
            PAYMENT_METHOD VARCHAR, 
            TOTAL_AMOUNT FLOAT
        )
        """).collect()
        
        # Create CUSTOMERS table
        session.sql("""
        CREATE OR REPLACE TABLE CUSTOMERS (
            CUSTOMER_ID VARCHAR,
            AGE BIGINT,
            CUSTOMER_SEGMENT VARCHAR,
            GENDER VARCHAR,
            LOCATION VARCHAR,
            SIGNUP_DATE DATE
        )
        """).collect()
        
        # Create FEEDBACK_RAW table
        session.sql("""
        CREATE OR REPLACE TABLE FEEDBACK_RAW (
            CHAT_DATE DATE,
            COMMENT VARCHAR,
            CUSTOMER_ID VARCHAR,
            FEEDBACK_ID VARCHAR,
            INTERNAL_ID BIGINT
        )
        """).collect()
        
        # Create stream for processing feedback
        session.sql("""
        CREATE OR REPLACE STREAM FEEDBACK_RAW_STREAM 
            ON TABLE FEEDBACK_RAW
            APPEND_ONLY = TRUE
        """).collect()
        
        # Create table for processed sentiment
        session.sql("""
        CREATE OR REPLACE TABLE FEEDBACK_SENTIMENT (
            FEEDBACK_ID VARCHAR,
            CHAT_DATE DATE,
            CUSTOMER_ID VARCHAR,
            INTERNAL_ID BIGINT,
            COMMENT VARCHAR,
            SENTIMENT FLOAT
        )
        """).collect()
        
        logger.info("Core data tables created successfully")
    
    @staticmethod
    def setup_staging_area(session: Session):
        """Setup staging area and data ingestion tracking."""
        session.sql(f"""
        CREATE OR REPLACE STAGE CSV
        DIRECTORY = (ENABLE = TRUE)
        URL = '{config.S3_URL}';
        """).collect()
        
        # Create tracking table for file ingestion
        session.sql("""
        CREATE OR REPLACE TABLE FILES_INGESTED (
            YEAR INT,
            MONTH INT,
            FILE_TYPE VARCHAR,
            FILE_NAME VARCHAR,
            STAGE_NAME VARCHAR,
            INGESTED BOOLEAN
        )
        """).collect()
        
        logger.info("Staging area and tracking tables created")


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def get_year_month_files(session: Session, stage_name: str, file_prefix: str) -> List[Tuple[int, int, str, str]]:
        """
        Extract year and month information from files in staging area.
        
        Args:
            session: Snowflake session
            stage_name: Name of the stage
            file_prefix: Prefix of files to look for
            
        Returns:
            List of tuples (year, month, file_name, stage_name)
        """
        list_files_query = f"LIST @{stage_name}"
        files = session.sql(list_files_query).collect()
        
        file_names = [file["name"].split("/")[-1] for file in files]
        file_pattern = re.compile(rf"{re.escape(file_prefix)}_(\d+)_(\d+)\.csv")
        
        results = []
        for file_name in file_names:
            match = file_pattern.match(file_name)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                results.append((year, month, file_name, stage_name))
        
        return sorted(results)
    
    @staticmethod
    def insert_file_tracking(session: Session, table: str, db: str, sc: str, files: List[Tuple]):
        """
        Track files for ingestion.
        
        Args:
            session: Snowflake session
            table: Table type (sales, feedback_raw, etc.)
            db: Database name
            sc: Schema name
            files: List of file tuples
        """
        for file in files:
            year, month, file_name, stage_name = file
            sql_cmd = f"""
                INSERT INTO {db}.{sc}.FILES_INGESTED
                (YEAR, MONTH, FILE_TYPE, FILE_NAME, STAGE_NAME, INGESTED)
                VALUES ('{year}', '{month}', '{table}', '{file_name}', '{stage_name}', False)
            """
            session.sql(sql_cmd).collect()
    
    @staticmethod
    def load_into_table(session: Session, table_name: str, file_name: str):
        """
        Load CSV file into Snowflake table.
        
        Args:
            session: Snowflake session
            table_name: Target table name
            file_name: Source file name with stage prefix
        """
        sql_cmd = f""" 
            COPY INTO {table_name}
            FROM {file_name}  
            FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"')  
            ON_ERROR = 'ABORT_STATEMENT';      
        """
        session.sql(sql_cmd).collect()


class DataTypeUtils:
    """Utility functions for data type operations."""
    
    @staticmethod
    def convert_numeric_columns(dataset_df, numeric_types):
        """
        Convert decimal columns to float for ML compatibility.
        
        Args:
            dataset_df: Snowpark DataFrame
            numeric_types: Tuple of numeric types
            
        Returns:
            DataFrame with converted numeric columns
        """
        # Convert decimal columns to float
        decimal_columns = [field.name for field in dataset_df.schema.fields
                          if isinstance(field.datatype, numeric_types)]
        
        for column_name in decimal_columns:
            dataset_df = dataset_df.with_column(
                column_name,
                F.col(column_name).cast("float")
            )
        
        return dataset_df


class FeatureStoreUtils:
    """Utility functions for Feature Store operations."""
    
    @staticmethod
    def create_entity_if_not_exists(fs, entity_name: str, join_keys: List[str], description: str):
        """
        Create entity in Feature Store if it doesn't exist.
        
        Args:
            fs: Feature Store instance
            entity_name: Name of the entity
            join_keys: List of join keys
            description: Entity description
            
        Returns:
            Entity instance
        """
        from snowflake.ml.feature_store import Entity
        
        try:
            # Check if entity exists
            existing_entities = json.loads(
                fs.list_entities().select(F.to_json(F.array_agg("NAME", True))).collect()[0][0]
            )
            
            if entity_name not in existing_entities:
                entity = Entity(
                    name=entity_name, 
                    join_keys=join_keys,
                    desc=description
                )
                fs.register_entity(entity)
                logger.info(f"Entity {entity_name} created")
            else:
                entity = fs.get_entity(entity_name)
                logger.info(f"Entity {entity_name} already exists")
                
            return entity
            
        except Exception as e:
            logger.error(f"Error creating entity {entity_name}: {str(e)}")
            raise
    
    @staticmethod
    def create_feature_view_if_not_exists(fs, entity, fv_name: str, fv_version: str, 
                                        labeled_df, timestamp_col: str, description: str):
        """
        Create Feature View if it doesn't exist.
        
        Args:
            fs: Feature Store instance
            entity: Entity instance
            fv_name: Feature View name
            fv_version: Feature View version
            labeled_df: Labeled DataFrame
            timestamp_col: Timestamp column name
            description: Feature View description
            
        Returns:
            Feature View instance
        """
        from snowflake.ml.feature_store import FeatureView
        
        try:
            feature_view = fs.get_feature_view(name=fv_name, version=fv_version)
            logger.info(f"Feature View {fv_name}_{fv_version} already exists")
        except:
            feature_view_instance = FeatureView(
                name=fv_name, 
                entities=[entity], 
                feature_df=labeled_df,
                timestamp_col=timestamp_col,
                refresh_freq=None,
                desc=description
            )
            
            feature_view = fs.register_feature_view(
                feature_view=feature_view_instance, 
                version=fv_version, 
                block=True
            )
            logger.info(f"Feature View {fv_name}_{fv_version} created")
        
        return feature_view


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def set_default_model(session: Session, mr_schema: str, sc: str, model_name: str, version_name: str):
        """
        Set default model version.
        
        Args:
            session: Snowflake session
            mr_schema: Model Registry schema name
            sc: Current schema name
            model_name: Model name
            version_name: Version name to set as default
        """
        session.sql(f'USE SCHEMA {mr_schema}').collect()
        session.sql(f'ALTER MODEL {model_name} SET DEFAULT_VERSION = {version_name};').collect()
        session.sql(f'USE SCHEMA {sc}').collect()
        logger.info(f"Default model set to: {version_name}")


class DateUtils:
    """Utility functions for date operations."""
    
    @staticmethod
    def create_version_name_from_date(timestamp) -> str:
        """
        Create version name from timestamp.
        
        Args:
            timestamp: Timestamp to convert
            
        Returns:
            Version name string
        """
        return f"v_{timestamp}".replace("-", "_")
    
    @staticmethod
    def format_duration(start_time: datetime, end_time: datetime) -> str:
        """
        Format duration between two timestamps.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Formatted duration string
        """
        duration = end_time - start_time
        return f"{duration.total_seconds():.1f}s"


class ValidationUtils:
    """Utility functions for data validation."""
    
    @staticmethod
    def validate_required_columns(df, required_columns: List[str]):
        """
        Validate that required columns exist in DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            ValueError if required columns are missing
        """
        df_columns = [field.name for field in df.schema.fields]
        missing_columns = [col for col in required_columns if col not in df_columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @staticmethod
    def validate_data_quality(df, min_records: int = 1):
        """
        Validate basic data quality requirements.
        
        Args:
            df: DataFrame to validate
            min_records: Minimum number of records required
            
        Raises:
            ValueError if data quality requirements are not met
        """
        record_count = df.count()
        
        if record_count < min_records:
            raise ValueError(f"Insufficient data: {record_count} records, minimum required: {min_records}")
        
        logger.info(f"Data quality validation passed: {record_count} records")


# Convenience functions for common operations
def log_execution_time(func):
    """Decorator to log execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = DateUtils.format_duration(start_time, end_time)
        logger.info(f"{func.__name__} completed in {duration}")
        return result
    return wrapper
