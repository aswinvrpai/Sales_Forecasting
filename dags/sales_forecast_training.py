from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

import logging

# Logger setup
logger = logging.getLogger(__name__)

sys.path.append('/usr/local/airflow/include/')

# Default arguments for the DAG
default_args = {
    'owner': 'aswinpai',
    'depends_on_past': False,
    'start_date': datetime(year=2025, month=8, day=1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'schedule_interval': '@weekly',
}

# Define the DAG
@dag(
    default_args=default_args,
    description='Sales Forecast Training Pipeline',
    tags=['sales', 'sales_forecast', 'training', 'ml'],
)

##
# DAG to extract, transform, and train a sales forecast model.
def sales_forecast_training():

    @task
    def extract_data_task():
        # Simulate data extraction
        from include.utils.data_generator import RealisticSalesDataGenerator

        data_output_dir = '/tmp/sales_data'
        os.makedirs(data_output_dir, exist_ok=True)
        generator = RealisticSalesDataGenerator(start_date="2023-01-01", end_date="2023-03-31")

        print("Generating sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)

        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Total files generated: {total_files}")

        for data_type, paths in file_paths.items():
            print(f"{data_type} files: len={len(paths)} files")

        return {
            'data_output_dir': data_output_dir,
            'total_files': total_files,
            'file_paths': file_paths
        }

    @task
    def validate_data(extracted_data):
        # Validate the extracted data

        # File paths
        file_paths = extracted_data['file_paths']

        # Sales files
        sales_files = file_paths.get('sales', [])
        if not sales_files:
            raise ValueError("No sales data files found!")
        logger.info(f"Found {len(sales_files)} sales data files.")

        # Only 10 files of Sales data expected
        sales_files = sales_files[:10]

        # Simple validation: check if files are non-empty
        for file in sales_files:
            if os.path.getsize(file) == 0:
                raise ValueError(f"Sales data file {file} is empty!")
        
        # Total Rows and Issues
        total_rows = 0
        issues_found = []

        # Loop through sales files to check for issues
        for i, sales_file in enumerate(sales_files):
            df = pd.read_parquet(sales_file)
            logger.info(f"Sales file {i} preview:\n{df.head(5)}")

            if df.empty:
                issues_found.append(f"Sales data file {sales_file} is empty!")
        
            required_columns = [
                'date', 
                'store_id', 
                'product_id', 
                'cost', 
                'quantity_sold', 
                'revenue'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues_found.append(f"Sales data file {sales_file} is missing columns: {missing_columns}")
            
            total_rows += len(df)

            # Validate Quantity Sold
            if (df['quantity_sold'] < 0).any():
                issues_found.append(f"Sales data file {sales_file} has negative quantity_sold values!")
            
            # Validate Revenue
            if (df['revenue'] < 0).any():
                issues_found.append(f"Sales data file {sales_file} has negative revenue values!")

        # Datatype checks
        datatype_list = ['promotions', 'customer_traffic', 'store_events']

        for datatype in datatype_list:
            datatype_files = file_paths.get(datatype, [])
            for datatype_file in datatype_files:
                df_dtype = pd.read_parquet(datatype_file)
                logger.info(f"{datatype} shape:{df_dtype.shape}")
                # logger.info(f"{datatype} columns:{df_dtype.columns.to_list()}")
            
        validation_report = {
            'total_files_validated': len(sales_files),
            'total_rows': total_rows,
            'issues_found': issues_found,
            'issues_count': len(issues_found),
            'file_paths': file_paths
        }

        if issues_found:
            logger.warning(f"Validation issues found: {issues_found}")
            for issue in issues_found:
                logger.error(issue)
            raise Exception("Data validation failed with issues.")
        else:
            logger.info("All sales data files passed validation.")

        # logger.info("Validation report:", validation_report.values())

        return validation_report
    
    def get_daily_sales_data(file_paths, max_files):
        # Get daily aggregated sales data from sales files

        sales_df = []
        sales_files = file_paths.get('sales', [])[:max_files]
        for sales_file in sales_files:
            df = pd.read_parquet(sales_file)
            sales_df.append(df)

            # Log after every 10 files
            if len(sales_df) % 10 == 0:
                logger.info(f"Loaded {len(sales_df)} sales files for training...")

        # Combine all sales data
        sales_data = pd.concat(sales_df, ignore_index=True)

        # Daily sales aggregation
        groupby_cols = ['date', 'category', 'store_id', 'product_id']
        daily_sales=sales_data.groupby(groupby_cols).agg({
            'quantity_sold':'sum', 
            'revenue':'sum','cost':'sum',
            'profit':'sum',
            'discount_percent':'mean',
            'unit_price':'mean'
        }).reset_index()

        # Rename columns of revenue to sales
        daily_sales.rename(columns={
            'revenue':'sales',
        }, inplace=True)

        return daily_sales
    
    def merge_promotions_data(daily_sales, file_paths):
        # Merge promotions data if available to Sales data

        if file_paths.get('promotions'):
            promotions_df = pd.concat([pd.read_parquet(f) for f in file_paths['promotions']], ignore_index=True)
            
            # Summarize promotions data
            promotion_summary = promotions_df.groupby(['date', 'product_id']).agg({
                'discount_percent':'max'
            }).reset_index()
            promotion_summary['has_promotion'] = 1

            # Merge promotions with daily sales
            daily_sales = daily_sales.merge(
                promotion_summary[['date', 'product_id', 'has_promotion']],
                how='left',
                on=['date', 'product_id']
            )
            daily_sales['has_promotion'] = daily_sales['has_promotion'].fillna(0).astype(int)
        
        return daily_sales
    
    def merge_customer_traffic_data(daily_sales, file_paths):
        # Merge customer traffic data if available to Sales data

        max_traffic_files = 10
        if file_paths.get('customer_traffic'):
            
            # Log only using limited files for traffic data
            logger.info(f"Customer traffic data of only {max_traffic_files} days is taken for merging to Sales data...")

            # Load customer traffic data for 10 days
            traffic_df = pd.concat([pd.read_parquet(f) for f in file_paths['customer_traffic'][:max_traffic_files]], ignore_index=True)
            
            # Rename columns
            traffic_df.rename(columns={'customer_traffic':'customer_count'}, inplace=True)
            
            # Summarize traffic data
            traffic_summary = traffic_df.groupby(['date', 'store_id']).agg({
                'customer_count':'sum', 'is_holiday':'max'
            }).reset_index()

            # Merge customer traffic with daily sales
            daily_sales = daily_sales.merge(
                traffic_summary,
                how='left',
                on=['date', 'store_id']
            )
            daily_sales['customer_count'] = daily_sales['customer_count'].fillna(0).astype(int)

        return daily_sales

    @task
    def train_models_task(validation_report):
        # Train sales forecast models

        # File paths
        file_paths = validation_report['file_paths']

        # Log training start
        logger.info("Starting model training...")

        # Limit to first 50 files for training
        max_files_to_use = 50
        
        # Combine all sales data from files;
        daily_sales = get_daily_sales_data(file_paths, max_files=max_files_to_use)
        logger.info(f"Total Daily sales data shape for training: {daily_sales.shape}")

        # Merge promotions data if available to Sales data
        logger.info(f"Merging Promotions data to Sales data if available...")
        daily_sales = merge_promotions_data(daily_sales, file_paths)
        
        # Merge customer traffic data if available to Sales data
        logger.info(f"Customer traffic data to Sales data if available...")
        daily_sales = merge_customer_traffic_data(daily_sales, file_paths)

        # Log final daily sales data shape
        logger.info(f"Final Daily sales data shape for training after merges: {daily_sales.shape}")
        logger.info(f"Daily sales data preview for training:{daily_sales.columns.to_list()}")

        # Aggregate to store level daily sales
        store_daily_sales = daily_sales.groupby(['date', 'store_id']).agg({
            'sales':'sum',
            'quantity_sold':'sum',
            'profit':'sum',
            'customer_count':'first',
            'is_holiday':'first',
            'has_promotion':'mean'
        }).reset_index()

        store_daily_sales['date'] = pd.to_datetime(store_daily_sales['date'])

    extracted_data = extract_data_task()
    validation_report = validate_data(extracted_data)
    train_models_task = train_models_task(validation_report)

sales_forecast_training_dag = sales_forecast_training()