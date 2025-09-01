#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for End-to-End ML Workflow with Observability.

This is the main entry point for running the complete customer churn detection
ML pipeline including data ingestion, feature engineering, model training,
and monitoring with observability.

Usage:
    python main_pipeline.py --mode [init|train|continuous|drift]
"""

import argparse
import sys
from datetime import datetime
from typing import Optional

# Snowflake imports
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.ml.registry import Registry

# Local imports
from config import config, logger
from data_ingestion import DataIngestionPipeline
from feature_engineering import FeatureEngineer, FeatureStoreManager
from labeling import ChurnLabeler
from model_training import ModelTrainer
from model_monitoring import ModelMonitor


class MLPipelineOrchestrator:
    """Main orchestrator for the ML pipeline."""
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.session = None
        self.db = None
        self.sc = None
        self.mr = None
        self.fs = None
        self.schema_names = {}
        
        # Initialize components
        self.data_ingestion = None
        self.feature_engineer = None
        self.labeler = None
        self.trainer = None
        self.monitor = None
        
        logger.info("ML Pipeline Orchestrator initialized")
    
    def run_fresh_setup(self):
        """Create all infrastructure from scratch - drops existing schemas and recreates everything."""
        logger.info("=" * 60)
        logger.info("STARTING FRESH INFRASTRUCTURE SETUP")
        logger.info("=" * 60)
        
        # Initialize Snowflake session
        logger.info("Initializing Snowflake session...")
        self.session = Session.builder.getOrCreate()
        logger.info("Snowflake session established")
        
        # Use configured database and schema
        self.db = config.DATABASE
        self.sc = config.SCHEMA
        logger.info(f"Using configured Database: {self.db}, Schema: {self.sc}")
        
        # Create database if it doesn't exist
        self.session.sql(f'CREATE OR REPLACE DATABASE {self.db}').collect()
        logger.info(f"Database {self.db} created")
        
        # Switch to the configured database and create schema
        self.session.sql(f'USE DATABASE {self.db}').collect()
        self.session.sql(f'CREATE SCHEMA IF NOT EXISTS {self.sc}').collect()
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        logger.info(f"Schema {self.sc} created/verified and set as current")
        
        # Initialize components
        self.data_ingestion = DataIngestionPipeline(self.session, self.db, self.sc)
        self.feature_engineer = FeatureEngineer(self.session, self.db, self.sc)
        self.labeler = ChurnLabeler(self.session, self.db, self.sc)
        
        # Setup ML infrastructure from scratch
        logger.info("Setting up ML infrastructure from scratch...")
        
        # Get schema names using config method
        self.schema_names = config.get_schema_names(self.db, self.sc)
        mr_schema = self.schema_names['model_registry']
        fs_schema = self.schema_names['feature_store']
        
        # Clean up and create Model Registry
        logger.info(f"Dropping and recreating schemas: {mr_schema}, {fs_schema}")
        self.session.sql(f'DROP SCHEMA IF EXISTS {mr_schema}').collect()
        self.session.sql(f'DROP SCHEMA IF EXISTS {fs_schema}').collect()
        
        # Create Model Registry
        self.session.sql(f'CREATE SCHEMA {mr_schema}').collect()
        self.mr = Registry(session=self.session, database_name=self.db, schema_name=mr_schema)
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        # Create Feature Store
        self.fs = FeatureStore(
            session=self.session, 
            database=self.db, 
            name=fs_schema,
            default_warehouse=config.WAREHOUSE, 
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST
        )
        
        # Initialize ML components
        self.trainer = ModelTrainer(self.session, self.mr, self.db, self.sc)
        self.monitor = ModelMonitor(self.session, self.mr, mr_schema, self.sc)
        
        logger.info(f"Model Registry created: {mr_schema}")
        logger.info(f"Feature Store created: {fs_schema}")
        
        # Initialize complete data environment
        env_result = self.data_ingestion.initialize_complete_environment()
        
        logger.info("✅ Fresh infrastructure setup completed successfully")
        logger.info(f"   Found {env_result['sales_files']} sales files and {env_result['feedback_files']} feedback files")
        
        return True
    
    def connect_to_existing_infrastructure(self):
        """Connect to existing infrastructure - just gets references to existing objects."""
        logger.info("=" * 60)
        logger.info("CONNECTING TO EXISTING INFRASTRUCTURE")
        logger.info("=" * 60)
        
        # Initialize Snowflake session
        logger.info("Initializing Snowflake session...")
        self.session = Session.builder.getOrCreate()
        logger.info("Snowflake session established")
        
        # Use configured database and schema
        self.db = config.DATABASE
        self.sc = config.SCHEMA
        logger.info(f"Using configured Database: {self.db}, Schema: {self.sc}")
        
        # Switch to the configured database and schema
        try:
            self.session.sql(f'USE DATABASE {self.db}').collect()
            self.session.sql(f'USE SCHEMA {self.sc}').collect()
            logger.info(f"Successfully switched to Database: {self.db}, Schema: {self.sc}")
        except Exception as e:
            logger.error(f"Failed to switch to Database {self.db}, Schema {self.sc}: {str(e)}")
            raise RuntimeError(f"Database {self.db} or Schema {self.sc} does not exist. Run with --mode init first.")
        
        # Initialize components
        self.data_ingestion = DataIngestionPipeline(self.session, self.db, self.sc)
        self.feature_engineer = FeatureEngineer(self.session, self.db, self.sc)
        self.labeler = ChurnLabeler(self.session, self.db, self.sc)
        
        # Connect to existing ML infrastructure
        logger.info("Connecting to existing ML infrastructure...")
        
        # Get schema names using config method
        self.schema_names = config.get_schema_names(self.db, self.sc)
        mr_schema = self.schema_names['model_registry']
        fs_schema = self.schema_names['feature_store']
        
        # Connect to existing Model Registry
        try:
            self.mr = Registry(session=self.session, database_name=self.db, schema_name=mr_schema)
            logger.info(f"Connected to existing Model Registry: {mr_schema}")
        except Exception as e:
            logger.error(f"Failed to connect to Model Registry {mr_schema}: {str(e)}")
            raise RuntimeError(f"Model Registry {mr_schema} does not exist. Run with --mode init first.")
        
        # Ensure we're in the correct schema
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        # Connect to existing Feature Store
        try:
            self.fs = FeatureStore(
                session=self.session, 
                database=self.db, 
                name=fs_schema,
                default_warehouse=config.WAREHOUSE, 
                creation_mode=CreationMode.FAIL_IF_NOT_EXIST
            )
            logger.info(f"Connected to existing Feature Store: {fs_schema}")
        except Exception as e:
            logger.error(f"Failed to connect to Feature Store {fs_schema}: {str(e)}")
            raise RuntimeError(f"Feature Store {fs_schema} does not exist. Run with --mode init first.")
        
        # Initialize ML components
        self.trainer = ModelTrainer(self.session, self.mr, self.db, self.sc)
        self.monitor = ModelMonitor(self.session, self.mr, mr_schema, self.sc)
        
        logger.info("✅ Successfully connected to existing infrastructure")
        
        return True
    
    def run_complete_initialization(self):
        """Run complete pipeline initialization including session, infrastructure, and data setup.
        
        This is a convenience method that calls run_fresh_setup() for backward compatibility.
        """
        return self.run_fresh_setup()
    
    def run_complete_initial_training(self, num_months: int = 4):
        """Run complete initial model training including data loading, feature engineering, training, and monitoring setup."""
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE INITIAL MODEL TRAINING")
        logger.info("=" * 60)
        
        # Create features and labels tables
        features_table = config.TABLES['features']
        labeled_table = config.TABLES['labeled']
        
        # Drop existing tables for fresh start
        self.session.sql(f'DROP TABLE IF EXISTS {features_table}').collect()
        self.session.sql(f'DROP TABLE IF EXISTS {labeled_table}').collect()
        
        # Load initial training data month by month, creating features for each month's latest timestamp
        logger.info(f"Loading initial training data ({num_months} months)...")
        
        for i in range(num_months):
            start_time = datetime.now()
            logger.info(f"Loading month {i+1}/{num_months}")
            
            # 1. Copy next monthly file
            if not self.data_ingestion.copy_next_file():
                break
                
            # 2. Process sentiment for new feedback
            self.data_ingestion.process_sentiment()
            
            # 3. Calculate features for the latest timestamp of the newly loaded month
            sales_df = self.session.table("SALES")
            latest_transaction = sales_df.select("TRANSACTION_DATE").agg({"TRANSACTION_DATE": "max"}).collect()[0][0]
            
            logger.info(f"Creating features for latest timestamp: {latest_transaction}")
            self.feature_engineer.create_customer_features(latest_transaction, features_table)
            
            # 4. Create churn labels
            self.labeler.create_churn_labels(features_table, labeled_table, config.CHURN_WINDOW)
            
            elapsed = datetime.now() - start_time
            logger.info(f"Month {i+1} completed in {elapsed.total_seconds():.1f}s")
        
        # Get data summary
        sales_count = self.session.table("SALES").count()
        customers_count = self.session.table("CUSTOMERS").count()
        feedback_count = self.session.table("FEEDBACK_SENTIMENT").count()
        
        logger.info("Initial data loading completed")
        logger.info(f"Data Summary:")
        logger.info(f"   Sales records: {sales_count:,}")
        logger.info(f"   Customers: {customers_count:,}")
        logger.info(f"   Feedback records: {feedback_count:,}")
        
        # Setup Feature Store
        labeled_df = self.session.table(f'{self.sc}.{labeled_table}')
        fs_manager = FeatureStoreManager(self.session, self.fs, self.db, self.sc)
        feature_view = fs_manager.setup_feature_store_complete(labeled_df)
        
        # Create training and validation datasets
        timestamps = self.session.table(labeled_table).select("TIMESTAMP").distinct().sort("TIMESTAMP").collect()
        
        if len(timestamps) < 3:
            raise ValueError("Need at least 3 timestamps for training and validation")
        
        # Use second timestamp for training, third for validation  
        training_timestamp = timestamps[1]["TIMESTAMP"]
        validation_timestamp = timestamps[2]["TIMESTAMP"]
        
        logger.info(f"Training timestamp: {training_timestamp}")
        logger.info(f"Validation timestamp: {validation_timestamp}")
        
        # Create datasets
        training_dataset = fs_manager.create_dataset_from_feature_store(feature_view, 'CHURN_TRAINING', training_timestamp)
        validation_dataset = fs_manager.create_dataset_from_feature_store(feature_view, 'CHURN_VALIDATION', validation_timestamp)
        
        logger.info(f"Training dataset: {training_dataset.count():,} records")
        logger.info(f"Validation dataset: {validation_dataset.count():,} records")
        
        # Train and register baseline model
        model_result = self.trainer.train_and_register_model(
            training_dataset,
            version_name="baseline",
            sample_data=training_dataset.select(config.feature_cols).limit(100),
            comment="Baseline model for customer churn detection",
            set_as_default=True
        )
        
        baseline_model = model_result['model_instance']
        
        # Evaluate on validation
        validation_metrics = self.trainer.evaluate_model_on_validation(baseline_model, validation_dataset)
        
        # Setup complete monitoring infrastructure
        self.monitor.setup_complete_monitoring_infrastructure()
        
        # Populate baseline predictions
        self.monitor.run_inference_and_store(baseline_model, training_dataset, 'customer_churn_baseline_predicted', 'CHURNED_PRED_BASE')
        self.monitor.run_inference_and_store(baseline_model, validation_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_PROD')
        self.monitor.run_inference_and_store(baseline_model, validation_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_BASE')
        self.monitor.run_inference_and_store(baseline_model, validation_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_RETRAIN')
        
        logger.info("✅ Complete initial training completed successfully")
        logger.info(f"   Baseline model F1 Score: {validation_metrics['validation_f1_score']:.4f}")
        
        return baseline_model, validation_metrics
    
    def run_continuous_learning(self, max_iterations: Optional[int] = None):
        """Run continuous learning pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING CONTINUOUS LEARNING PIPELINE")
        logger.info("=" * 60)
        
        iteration = 1
        continue_processing = True
        
        while continue_processing and (max_iterations is None or iteration <= max_iterations):
            logger.info(f"\n🔄 Processing Iteration {iteration}")
            
            try:
                continue_processing = self.process_monthly_data()
                if continue_processing:
                    iteration += 1
                else:
                    logger.info("🎯 All data processed!")
                    
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration}: {str(e)}")
                break
        
        logger.info(f"✅ Pipeline completed after {iteration-1} iterations")
        
        # Final summary
        self._print_final_summary()
        
        return iteration - 1
    
    def process_monthly_data(self) -> bool:
        """Process new monthly data and retrain if needed."""
        start_time = datetime.now()
        
        logger.info("🔄 Processing new monthly data...")
        
        # Step 1: Load new data
        if not self.data_ingestion.copy_next_file():
            logger.info("❌ No more data files to process")
            return False
        
        # Step 2: Process sentiment
        self.data_ingestion.process_sentiment()
        
        # Step 3: Create features for new data
        sales_df = self.session.table("SALES")
        latest_timestamp = sales_df.select(F.max("TRANSACTION_DATE")).collect()[0][0]
        
        logger.info(f"📊 Creating features for timestamp: {latest_timestamp}")
        features_table = config.TABLES['features']
        labeled_table = config.TABLES['labeled']
        
        self.feature_engineer.create_customer_features(latest_timestamp, features_table)
        
        # Step 4: Update labels
        self.labeler.create_churn_labels(features_table, labeled_table, config.CHURN_WINDOW)
        
        # Step 5: Create new dataset and run inference
        latest_labeled_timestamp = self.session.table(labeled_table).select(F.max("TIMESTAMP")).collect()[0][0]
        
        fs_manager = FeatureStoreManager(self.session, self.fs, self.db, self.sc)
        feature_view = self.fs.get_feature_view(name="FV_CUSTOMER_CHURN", version="V_1")
        # Ensure we're in the correct schema after Feature Store operation
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        
        date_name = f"v_{latest_labeled_timestamp}".replace("-", "_")
        new_dataset = fs_manager.create_dataset_from_feature_store(feature_view, f'CHURN_{date_name}', latest_labeled_timestamp)
        
        # Step 6: Run inference and get performance
        baseline_model = self.mr.get_model(config.MODEL_NAME).version("baseline")
        # Ensure we're in the correct schema after Model Registry operation
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        self.monitor.run_inference_and_store(baseline_model, new_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_BASE')
        self.labeler.update_labels_with_new_data("CUSTOMER_CHURN_PREDICTED_PROD2", config.CHURN_WINDOW)
        
        # Get performance metrics
        baseline_f1 = self.monitor.get_model_performance_sklearn('CHURNED_PRED_BASE')
        production_f1 = self.monitor.get_model_performance_sklearn('CHURNED_PRED_PROD')
        
        logger.info("📈 Model Performance Metrics:")
        logger.info(f"   Baseline F1:   {baseline_f1:.4f}")
        logger.info(f"   Production F1: {production_f1:.4f}")
        
        # Step 7: Retrain if performance drops
        retrained_f1 = 0.0
        best_model = "baseline"
        
        if production_f1 < config.RETRAIN_THRESHOLD:
            logger.info(f"🚨 Performance dropped below {config.RETRAIN_THRESHOLD}, retraining model...")
            
            # Train and register new model
            new_model_result = self.trainer.train_and_register_model(
                new_dataset,
                version_name=date_name,
                sample_data=new_dataset.select(config.feature_cols).limit(100),
                comment=f"Retrained model for {latest_labeled_timestamp}"
            )
            
            retrained_model = new_model_result['model_instance']
            
            logger.info(f"✅ New model {date_name} trained:")
            logger.info(f"   Train F1: {new_model_result['metrics']['train_f1_score']:.4f}")
            logger.info(f"   Test F1: {new_model_result['metrics']['test_f1_score']:.4f}")
            
            # Test new model
            self.monitor.run_inference_and_store(retrained_model, new_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_RETRAIN')
            retrained_f1 = self.monitor.get_model_performance_sklearn('CHURNED_PRED_RETRAIN')
            
            logger.info(f"   Validation F1: {retrained_f1:.4f}")
            
            # Choose best model
            if retrained_f1 > max(baseline_f1, production_f1):
                mr_schema = self.schema_names['model_registry']
                self.session.sql(f'USE SCHEMA {mr_schema}').collect()
                self.session.sql(f'ALTER MODEL {config.MODEL_NAME} SET DEFAULT_VERSION = {date_name};').collect()
                self.session.sql(f'USE SCHEMA {self.sc}').collect()
                best_model = date_name
        
        # Run production inference
        prod_model = self.mr.get_model(config.MODEL_NAME).default
        # Ensure we're in the correct schema after Model Registry operation
        self.session.sql(f'USE SCHEMA {self.sc}').collect()
        self.monitor.run_inference_and_store(prod_model, new_dataset, 'CUSTOMER_CHURN_PREDICTED_PROD2', 'CHURNED_PRED_PROD')
        
        elapsed = datetime.now() - start_time
        logger.info(f"✅ Monthly processing completed in {elapsed.total_seconds():.1f}s")
        logger.info(f"   Best model: {best_model}")
        
        return True
    
    def run_drift_simulation(self):
        """Run data drift simulation."""
        logger.info("=" * 60)
        logger.info("STARTING DATA DRIFT SIMULATION")
        logger.info("=" * 60)
        
        # Setup drift data
        new_sales_count, new_feedback_count = self.data_ingestion.setup_drift_data()
        
        logger.info(f"Registered {new_sales_count} new sales files and {new_feedback_count} new feedback files")
        logger.info("These files contain:")
        logger.info("- Different customer segments (demographic drift)")
        logger.info("- New purchasing patterns (behavioral drift)")
        logger.info("- Different sentiment distributions (feature drift)")
        
        # Process drift data
        self.run_continuous_learning(max_iterations=5)  # Limit iterations for drift simulation
        
        logger.info("✅ Data drift simulation completed")
    
    def _print_final_summary(self):
        """Print final pipeline summary."""
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL SUMMARY")
        logger.info("="*60)
        
        
        # Check model versions
        try:
            models = self.session.sql(f"""
                SELECT VERSION_NAME, CREATION_TIME 
                FROM {self.schema_names['model_registry']}.INFORMATION_SCHEMA.ML_MODELS 
                WHERE MODEL_NAME = '{config.MODEL_NAME}'
                ORDER BY CREATION_TIME
            """).collect()
            
            logger.info("\n🤖 Models in Registry:")
            for model in models:
                logger.info(f"   - {model['VERSION_NAME']} (created: {model['CREATION_TIME']})")
        except:
            logger.info("   Model information not available")
        
        logger.info("\n🎯 Workflow completed successfully!")
        logger.info("="*60)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='End-to-End ML Pipeline with Observability')
    parser.add_argument('--mode', choices=['init', 'train', 'continuous', 'drift', 'all'], 
                       default='init', 
                       help='Pipeline mode to run: '
                            'init (create infrastructure from scratch), '
                            'train (connect to existing infrastructure + run initial training), '
                            'continuous (connect to existing + run continuous learning), '
                            'drift (connect to existing + run drift simulation), '
                            'all (fresh setup + all training steps)')
    parser.add_argument('--months', type=int, default=4, 
                       help='Number of months for initial training')
    parser.add_argument('--max-iterations', type=int, 
                       help='Maximum iterations for continuous learning')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        pipeline = MLPipelineOrchestrator()
        
        if args.mode == 'init':
            # Run fresh initialization - creates everything from scratch
            pipeline.run_fresh_setup()
            
        elif args.mode == 'train':
            # Connect to existing infrastructure and run initial training
            pipeline.connect_to_existing_infrastructure()
            pipeline.run_complete_initial_training(num_months=args.months)
            
        elif args.mode == 'continuous':
            # Connect to existing infrastructure and run continuous learning
            pipeline.connect_to_existing_infrastructure()
            pipeline.run_continuous_learning(max_iterations=args.max_iterations)
            
        elif args.mode == 'drift':
            # Connect to existing infrastructure and run drift simulation
            pipeline.connect_to_existing_infrastructure()
            pipeline.run_drift_simulation()
        
        elif args.mode == 'all':
            # Run all steps - fresh setup first, then all training steps
            pipeline.run_fresh_setup()
            pipeline.run_complete_initial_training(num_months=args.months)
            pipeline.run_continuous_learning(max_iterations=args.max_iterations)
            pipeline.run_drift_simulation()
        

        logger.info("🎉 Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
