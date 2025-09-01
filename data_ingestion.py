"""
Data ingestion module for the End-to-End ML Workflow with Observability.

This module handles data loading, processing, and sentiment analysis
for the customer churn prediction pipeline.
"""

import re
from datetime import datetime
from typing import Tuple, List
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.cortex import sentiment

from config import config, logger


class DataIngestionPipeline:
    """Main class for handling data ingestion operations."""
    
    def __init__(self, session: Session, db: str, sc: str):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            session: Snowflake session
            db: Database name
            sc: Schema name
        """
        self.session = session
        self.db = db
        self.sc = sc
    
    def initialize_complete_environment(self) -> dict:
        """
        Initialize the complete data environment including schemas, tables, staging, and file discovery.
        
        Returns:
            Dictionary with setup results
        """
        logger.info("Initializing complete data environment...")
        
        # Create and use dedicated schema
        self.session.sql(f'CREATE OR REPLACE SCHEMA {self.sc}').collect()
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        self.session.sql('CREATE OR REPLACE STAGE ML_STAGE').collect()
        
        logger.info(f"Environment configured - Schema: {self.sc}")
        
        # Create all core tables
        logger.info("Creating core data tables...")
        
        # Create SALES table
        self.session.sql("""
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
        self.session.sql("""
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
        self.session.sql("""
        CREATE OR REPLACE TABLE FEEDBACK_RAW (
            CHAT_DATE DATE,
            COMMENT VARCHAR,
            CUSTOMER_ID VARCHAR,
            FEEDBACK_ID VARCHAR,
            INTERNAL_ID BIGINT
        )
        """).collect()
        
        # Create stream for processing feedback
        self.session.sql("""
        CREATE OR REPLACE STREAM FEEDBACK_RAW_STREAM 
            ON TABLE FEEDBACK_RAW
            APPEND_ONLY = TRUE
        """).collect()
        
        # Create table for processed sentiment
        self.session.sql("""
        CREATE OR REPLACE TABLE FEEDBACK_SENTIMENT (
            FEEDBACK_ID VARCHAR,
            CHAT_DATE DATE,
            CUSTOMER_ID VARCHAR,
            INTERNAL_ID BIGINT,
            COMMENT VARCHAR,
            SENTIMENT FLOAT
        )
        """).collect()
        
        # Setup staging area and tracking
        self.session.sql(f"""
        CREATE OR REPLACE STAGE CSV
        DIRECTORY = (ENABLE = TRUE)
        URL = '{config.S3_URL}';
        """).collect()
        
        # Create tracking table for file ingestion
        self.session.sql("""
        CREATE OR REPLACE TABLE FILES_INGESTED (
            YEAR INT,
            MONTH INT,
            FILE_TYPE VARCHAR,
            FILE_NAME VARCHAR,
            STAGE_NAME VARCHAR,
            INGESTED BOOLEAN
        )
        """).collect()
        
        # Discover and register files
        stage_name = config.STAGE_NAME
        sales_files = self._get_year_month_files(stage_name, 'sales')
        feedback_files = self._get_year_month_files(stage_name, 'feedback_raw')
        
        # Track files for ingestion
        self._insert_file_tracking('sales', sales_files)
        self._insert_file_tracking('feedback_raw', feedback_files)
        
        # Load static customer data
        customer_file = f'@{config.STAGE_NAME}/customers.csv'
        self._load_into_table('CUSTOMERS', customer_file)
        
        logger.info("Core data tables created successfully")
        logger.info("Staging area and tracking tables created")
        logger.info(f"Found {len(sales_files)} sales files and {len(feedback_files)} feedback files")
        logger.info("Customer data loaded")
        
        # Get derived schema names
        schema_names = {
            'model_registry': f'{self.sc}_MODEL_REGISTRY'.replace('"', ''),
            'feature_store': f'{self.sc}_FEATURE_STORE'.replace('"', '')
        }
        
        return {
            'schema_names': schema_names,
            'sales_files': len(sales_files),
            'feedback_files': len(feedback_files)
        }
    
    def _get_year_month_files(self, stage_name: str, file_prefix: str) -> List[Tuple]:
        """Extract year and month information from files in staging area."""
        list_files_query = f"LIST @{stage_name}"
        files = self.session.sql(list_files_query).collect()
        
        file_names = [file["name"].split("/")[-1] for file in files]
        file_pattern = re.compile(rf"{re.escape(file_prefix)}_(\d+)_(\d+)\.csv")
        
        results = []
        for file_name in file_names:
            match = file_pattern.match(file_name)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                results.append((year, month, file_name, stage_name))
        
        return sorted(results)
    
    def _insert_file_tracking(self, table: str, files: List[Tuple]):
        """Track files for ingestion."""
        for file in files:
            year, month, file_name, stage_name = file
            self.session.sql(f"""
                INSERT INTO FILES_INGESTED
                (YEAR, MONTH, FILE_TYPE, FILE_NAME, STAGE_NAME, INGESTED)
                VALUES ('{year}', '{month}', '{table}', '{file_name}', '{stage_name}', False)
            """).collect()
    
    def _load_into_table(self, table_name: str, file_name: str):
        """Load CSV file into Snowflake table."""
        self.session.sql(f""" 
            COPY INTO {table_name}
            FROM {file_name}  
            FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"')  
            ON_ERROR = 'ABORT_STATEMENT';      
        """).collect()
    
    def load_monthly_data_batch(self, num_months: int = 4) -> dict:
        """
        Load initial training data for specified number of months.
        
        Args:
            num_months: Number of months to load for initial training
            
        Returns:
            Dictionary with data summary statistics
        """
        logger.info(f"Loading initial training data ({num_months} months)...")
        
        for i in range(num_months):
            start_time = datetime.now()
            logger.info(f"Loading month {i+1}/{num_months}")
            
            if not self.copy_next_file():
                break
                
            # Process sentiment for new feedback
            self.process_sentiment()
            
            elapsed = datetime.now() - start_time
            logger.info(f"Month {i+1} completed in {elapsed.total_seconds():.1f}s")
        
        # Get data summary
        sales_count = self.session.table("SALES").count()
        customers_count = self.session.table("CUSTOMERS").count()
        feedback_count = self.session.table("FEEDBACK_SENTIMENT").count()
        
        summary = {
            'sales_records': sales_count,
            'customers': customers_count,
            'feedback_records': feedback_count
        }
        
        logger.info("Initial data loading completed")
        logger.info(f"Data Summary: {summary}")
        
        return summary
    
    def copy_next_file(self) -> bool:
        """
        Copy the next unprocessed file from staging to tables.
        
        Returns:
            True if files were loaded, False if no unprocessed files found
        """
        files_df = self.session.table("FILES_INGESTED")
        
        # Get next sales file
        sales_file = files_df.filter(
            (F.col("file_type") == 'sales') & (F.col("ingested") == False)
        ).select("year", "month", "file_name", "stage_name").order_by("year", "month").limit(1)
        
        sales_pd = sales_file.to_pandas()
        if sales_pd.empty:
            logger.info("No unprocessed sales files found")
            return False
        
        # Load sales data
        year, month = int(sales_pd.YEAR[0]), int(sales_pd.MONTH[0])
        file_name, stage_name = sales_pd.FILE_NAME[0], sales_pd.STAGE_NAME[0]
        
        self._load_into_table("SALES", f'@{stage_name}/{file_name}')
        
        # Mark sales file as processed
        self.session.sql(f"""
            UPDATE FILES_INGESTED
            SET INGESTED = TRUE
            WHERE FILE_NAME = '{file_name}' AND FILE_TYPE = 'sales'
        """).collect()
        
        # Load corresponding feedback file
        feedback_file = files_df.filter(
            (F.col("file_type") == 'feedback_raw') & 
            (F.col("ingested") == False) &
            (F.col("YEAR") == year) & 
            (F.col("MONTH") == month)
        ).limit(1)
        
        if feedback_file.count() > 0:
            feedback_pd = feedback_file.to_pandas()
            feedback_name = feedback_pd.FILE_NAME[0]
            
            self._load_into_table("FEEDBACK_RAW", f'@{stage_name}/{feedback_name}')
            
            # Mark feedback file as processed
            self.session.sql(f"""
                UPDATE FILES_INGESTED
                SET INGESTED = TRUE
                WHERE FILE_NAME = '{feedback_name}' AND FILE_TYPE = 'feedback_raw'
            """).collect()
        
        logger.info(f"Loaded data for {year}-{month:02d}")
        return True
    
    def process_sentiment(self):
        """Process sentiment for new feedback using Cortex AI."""
        feedback_stream_df = self.session.table("FEEDBACK_RAW_STREAM")
        
        if feedback_stream_df.count() > 0:
            cols = ['FEEDBACK_ID', 'CHAT_DATE', 'CUSTOMER_ID', 'INTERNAL_ID', 'COMMENT']
            feedback_sentiment_df = feedback_stream_df.select(cols).with_columns(
                ["SENTIMENT"], [sentiment(F.col("COMMENT"))]
            )
            feedback_sentiment_df.write.mode("append").save_as_table("FEEDBACK_SENTIMENT")
            logger.info("Sentiment processed for new feedback")
    
    def setup_drift_data(self) -> Tuple[int, int]:
        """
        Register new data files for drift simulation.
        
        Returns:
            Tuple of (new_sales_files, new_feedback_files)
        """
        logger.info("Setting up drift data...")
        
        stage_name = config.STAGE_NAME
        
        # Load new customers with different demographic patterns
        logger.info("Loading new customers with different demographics...")
        new_customer_file = f'@{stage_name}/new_customers.csv'
        self._load_into_table("CUSTOMERS", new_customer_file)
        logger.info("New customers loaded - demographic drift introduced")
        
        # Register new sales files (with different purchasing patterns)
        new_sales_files = self._get_year_month_files(stage_name, 'new_sales')
        self._insert_file_tracking('sales', new_sales_files)
        
        # Register new feedback files (with different sentiment patterns)  
        new_feedback_files = self._get_year_month_files(stage_name, 'new_feedback_raw2')
        self._insert_file_tracking('feedback_raw', new_feedback_files)
        
        logger.info(f"Registered {len(new_sales_files)} new sales files and {len(new_feedback_files)} new feedback files")
        
        return len(new_sales_files), len(new_feedback_files)
