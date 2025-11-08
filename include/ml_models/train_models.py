from datetime import datetime
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import yaml

from utils.ml_utils import MlFlowManager
from feature_engineering.feature_pipeline import FeatureEngineeringPipeline
from data_validation.validators import DataValidator
import pandas as pd
import numpy as np
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# Model training class
class ModelTrainer:

    # Initialize with configuration
    # Load settings from YAML config
    def __init__(self, config_path: str = '/usr/local/airflow/include/config/model_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model_config = self.config.get('models', {})
        self.training_config = self.config.get('training', {})

        # Initialize MLflow manager
        self.mlflow_manager = MlFlowManager(config_path)

        # Initialize Feature Engineer
        self.feature_engineer = FeatureEngineeringPipeline(config_path)

        # Data validator
        self.data_validator = DataValidator(config_path)

        # Models;
        self.models = {}

        # Scalers;
        self.scalers = {}

        # Encoders;
        self.encoders = {}

    # Prepare data for training
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sales', date_col: str = 'date', group_cols: Optional[list[str]] = None, categorical_cols: Optional[list[str]] = None) -> pd.DataFrame:
        
        # Logging;
        logger.info("Starting data preparation")

        # Required columns;
        required_columns = ['date', target_col]
        if group_cols:
            required_columns.extend(group_cols)
        
        # Raise error if required columns are missing;
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Features;
        df_features = self.feature_engineer.create_all_features(df, target_col=target_col, date_col=date_col, group_cols=group_cols, categorical_cols=categorical_cols)

        # Chronological sort for time series;
        df_sorted = df_features.sort_values(by=date_col).reset_index(drop=True)

        # Train Size
        train_size = int(len(df_sorted) * (1- self.training_config.get('test_size', 0.8) - self.training_config.get('validation_size', 0.1)))
        validation_size = int(len(df_sorted) * self.training_config.get('validation_size', 0.1))

        # Train Data;
        df_train = df_sorted.iloc[:train_size]
        logger.info(f"Prepared training data with {len(df_train)} records")
        
        # Validation Data;
        df_validation = df_sorted.iloc[train_size:train_size + validation_size]
        logger.info(f"Prepared validation data with {len(df_validation)} records")

        # Test Data;
        df_test = df_sorted.iloc[train_size + validation_size:]
        logger.info(f"Prepared test data with {len(df_test)} records")

        # Train Data
        df_train = df_train.dropna(subset=target_col)
        df_validation = df_validation.dropna(subset=target_col)
        df_test = df_test.dropna(subset=target_col)
        logger.info(f"Dropped NA values from target column '{target_col}'")

        return df_train, df_validation, df_test
    
    # Preprocess features
    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,target_col: str = 'sales', exclude_cols: Optional[list[str]] = None):

        # Logging;
        logger.info("Starting feature preprocessing")

        # Feature columns;
        feature_cols = [col for col in train_df.columns if col not in (exclude_cols or []) + [target_col]]

        # Train Features and Target;
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].values

        # Validation Features and Target;
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].values

        # Test Features and Target;
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].values

        # Encode categorical features;
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:, col] = self.encoders[col].transform(X_train[col].astype(str))
            
            X_val.loc[:, col] = self.encoders[col].transform(X_val[col].astype(str))
            X_test.loc[:, col] = self.encoders[col].transform(X_test[col].astype(str))

        # Scale numerical features;
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Dataframes after preprocessing;
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    # Calculate evaluation metrics function;
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2_Score': r2
        }

        return metrics

    # Train XGBoost model with optional hyperparameter tuning
    def train_xgboost_model(self, X_train, y_train, X_val, y_val, use_optuna: bool = True):
        import xgboost as xgb
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error
        import optuna

        # Logging;
        logger.info("Starting XGBoost model training")

        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }

                # Earlier stopping rounds;
                params['earlier_stopping_rounds'] = 50

                # Train model;
                model = XGBRegressor(**params, random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))

                return rmse

            # Optimize hyperparameters;
            study = optuna.create_study(direction='minimize', 
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=self.training_config.get('optuna_trials', 50))

            # Best hyperparameters;
            best_params = study.best_params
            logger.info(f"Best hyperparameters from Optuna: {best_params}")

            # Model with best hyperparameters;
            model = XGBRegressor(**best_params, random_state=42)
        else:
            model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)

        # Final training with early stopping;
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)

        self.models['xgboost'] = model

        return model
    
    # Train all models;
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'sales', use_optuna: bool = True):

        # Logging;
        logger.info("Starting model training process")

        results = {}

        # Start MLflow run;
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
        )

        # Model Run;
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col=target_col, exclude_cols=['date']
            )

            # ML Flow logging;
            self.mlflow_manager.log_params({
                "feature_columns": self.feature_cols,
                "num_train_samples": len(X_train),
                "num_val_samples": len(X_val),
                "num_test_samples": len(X_test)
            })

            # Train XGBoost Model;
            xgb_model = self.train_xgboost_model(X_train, y_train, X_val, y_val, use_optuna=use_optuna)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)

            # Log metrics in ML Flow;
            self.mlflow_manager.log_metrics({
                f"xgb_{key}": value for key, value in xgb_metrics.items()
            })

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            self.mlflow_manager.end_run(failed=True)




        pass
        