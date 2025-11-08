import os
import yaml
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from service_discovery import get_mlflow_endpoint, get_minio_endpoint

logger = logging.getLogger(__name__)

# MLflow Manager class
# Handles MLflow configuration and setup
class MlFlowManager:
    def __init__(self,config_path: str = '/usr/local/airflow/include/config/mlflow_config.yaml'):

        # Load MLflow configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Extract MLflow settings
        self.mlflow_config = self.config.get('mlflow', {})
        self.tracking_uri = self.mlflow_config.get('tracking_uri', 'http://localhost:5000')
        self.experiment_name = self.mlflow_config.get('experiment_name', 'default_experiment')
        self.registry_name = self.mlflow_config.get('registry_name', 'default_registry')

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set or create MLflow experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"Experiment {self.experiment_name} not found. Creating a new experiment.")
            if 'mlflow' in self.tracking_uri:
                self.tracking_uri = 'http://localhost:5001'
                mlflow.set_tracking_uri(self.tracking_uri)
                os.setenv('MLFLOW_TRACKING_URI', self.tracking_uri)
                logger.info(f"Set MLFLOW_TRACKING_URI to {self.tracking_uri}")

                try:
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e:
                    logger.error(f"Failed to create experiment {self.experiment_name}: {e}")
        
        # Configure MinIO endpoint for MLflow S3 storage
        os.environ['MLFLOW_S3_ENDPOINT_URI'] = get_minio_endpoint()
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')

        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)