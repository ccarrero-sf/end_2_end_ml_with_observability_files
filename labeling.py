"""
Labeling module for the End-to-End ML Workflow with Observability.

This module handles the creation of churn labels based on customer behavior
and future purchase patterns.
"""

from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from config import config, logger


class ChurnLabeler:
    """Main class for creating churn labels."""
    
    def __init__(self, session: Session, db: str, sc: str):
        """
        Initialize the churn labeler.
        
        Args:
            session: Snowflake session
            db: Database name
            sc: Schema name
        """
        self.session = session
        self.db = db
        self.sc = sc
    
    def create_churn_labels(self, features_table: str, output_table: str, churn_days: int):
        """
        Label customers as churned based on future purchase behavior.
        
        Args:
            features_table: Table containing customer features
            output_table: Output table for labeled data
            churn_days: Number of days to define churn window
        """
        logger.info(f"Creating churn labels with {churn_days} day window")
        
        # Load features and sales data
        features_df = self.session.table(features_table)
        sales_df = self.session.table("SALES")
        
        # Validate minimum data requirements
        if features_df.count() < 10:
            raise ValueError("Insufficient feature data for labeling")
        if sales_df.count() < 10:
            raise ValueError("Insufficient sales data for labeling")
        
        # Filter sales to retain only customer ID and transaction date
        sales_filtered = sales_df.select(F.col("CUSTOMER_ID"), F.col("TRANSACTION_DATE"))
        
        # Find next transaction for each customer after their feature timestamp
        next_transaction_df = features_df.join(
            sales_filtered.select("CUSTOMER_ID", "TRANSACTION_DATE"),
            "CUSTOMER_ID",
            "left"
        ).filter(
            F.col("TRANSACTION_DATE") > F.col("LAST_PURCHASE_DATE")
        ).group_by("CUSTOMER_ID", "TIMESTAMP").agg(
            F.min("TRANSACTION_DATE").alias("NEXT_TRANSACTION_DATE")
        )
        
        # Create labeled dataset
        labeled_df = features_df.join(
            next_transaction_df, 
            ["CUSTOMER_ID", "TIMESTAMP"], 
            "left"
        ).select(
            features_df["*"],
            F.when(
                (F.col("NEXT_TRANSACTION_DATE").is_null()) |
                ((F.col("NEXT_TRANSACTION_DATE") - F.col("LAST_PURCHASE_DATE")) > churn_days),
                1
            ).otherwise(0).alias("CHURNED"),
            F.col("NEXT_TRANSACTION_DATE")
        )
        
        # Save labeled dataset
        labeled_df.write.mode("overwrite").save_as_table(output_table)
        
        # Log label distribution
        try:
            distribution_stats = labeled_df.group_by("TIMESTAMP").agg(
                F.sum(F.when(F.col("CHURNED") == 0, 1).otherwise(0)).alias("NOT_CHURNED"),
                F.sum(F.when(F.col("CHURNED") == 1, 1).otherwise(0)).alias("CHURNED")
            ).sort("TIMESTAMP").collect()
            
            logger.info("Label Distribution by Timestamp:")
            for row in distribution_stats:
                timestamp = row["TIMESTAMP"]
                not_churned = row["NOT_CHURNED"]
                churned = row["CHURNED"]
                total = not_churned + churned
                churn_rate = churned / total * 100 if total > 0 else 0
                logger.info(f"  {timestamp}: {not_churned} not churned, {churned} churned ({churn_rate:.1f}% churn rate)")
                
        except Exception as e:
            logger.warning(f"Could not calculate label distribution: {str(e)}")
        
        logger.info(f"Labels created for churn window: {churn_days} days")
    
    def update_labels_with_new_data(self, table_name: str, churn_days: int):
        """
        Update labels with new transaction data.
        
        Args:
            table_name: Table containing existing labels
            churn_days: Number of days to define churn window
        """
        logger.info("Updating labels with new transaction data...")
        
        # Load existing data
        baseline_df = self.session.table(table_name)
        sales_df = self.session.table("SALES")
        
        # Filter sales to retain only customer ID and transaction date
        sales_filtered = sales_df.select(F.col("CUSTOMER_ID"), F.col("TRANSACTION_DATE"))
        
        # Find the next transaction date for each (CUSTOMER_ID, TIMESTAMP)
        next_transaction_df = (
            baseline_df
            .join(sales_filtered, "CUSTOMER_ID", "left")
            .filter(F.col("TRANSACTION_DATE") > F.col("LAST_PURCHASE_DATE"))
            .group_by(F.col("CUSTOMER_ID"), F.col("TIMESTAMP"))
            .agg(F.min("TRANSACTION_DATE").alias("NEXT_TX_DATE"))
        )
        
        # Join back with the baseline dataset to compute CHURNED
        final_df = (
            baseline_df
            .join(next_transaction_df, ["CUSTOMER_ID", "TIMESTAMP"], "left")
            .select(
                baseline_df["CUSTOMER_ID"],
                baseline_df["TIMESTAMP"],
                next_transaction_df["NEXT_TX_DATE"],
                F.when(
                    next_transaction_df["NEXT_TX_DATE"].is_null() |
                    ((next_transaction_df["NEXT_TX_DATE"] - baseline_df["LAST_PURCHASE_DATE"]) > churn_days),
                    1
                ).otherwise(0).alias("CHURNED")
            )    
            .with_column_renamed("NEXT_TX_DATE", "NEXT_TRANSACTION_DATE")
        )
        
        # Save temporary updates
        final_df.write.mode("overwrite").save_as_table('temp_updates')
        
        # Update the original table
        update_statement = f"""
            UPDATE {table_name} c
            SET CHURNED = t.CHURNED,
                NEXT_TRANSACTION_DATE = t.NEXT_TRANSACTION_DATE
            FROM temp_updates t
            WHERE c.CUSTOMER_ID = t.CUSTOMER_ID AND
                c.TIMESTAMP = t.TIMESTAMP
        """
        
        self.session.sql(update_statement).collect()
        
        logger.info("Labels updated with new transaction data")
