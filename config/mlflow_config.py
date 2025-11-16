"""
MLflow Configuration for Databricks
Handles connection to Databricks workspace and experiment tracking
"""

import os
from pathlib import Path
from typing import Optional
import mlflow
from dotenv import load_dotenv


def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> str:
    """
    Configure MLflow to use Databricks tracking server.
    
    Args:
        experiment_name: Name of the MLflow experiment. 
                        If None, uses value from .env or config.yaml
        tracking_uri: MLflow tracking URI. 
                     If None, uses 'databricks' (from .env)
    
    Returns:
        experiment_id: ID of the MLflow experiment
    
    Usage:
        >>> from config.mlflow_config import setup_mlflow
        >>> exp_id = setup_mlflow()
        >>> with mlflow.start_run():
        >>>     mlflow.log_param("learning_rate", 0.001)
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("⚠️  Warning: .env file not found. Using environment variables.")
    
    # Get Databricks credentials
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    
    # Validate credentials
    if not databricks_token:
        raise ValueError(
            "DATABRICKS_TOKEN not found in environment. "
            "Please set it in .env file or as environment variable."
        )
    
    # Set Databricks environment variables for MLflow
    if databricks_host:
        os.environ["DATABRICKS_HOST"] = databricks_host
    os.environ["DATABRICKS_TOKEN"] = databricks_token
    
    # Set tracking URI
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ MLflow tracking URI set to: {tracking_uri}")
    
    # Set experiment name
    if experiment_name is None:
        experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", 
            "ingredients_classification"
        )
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"❌ Error setting up MLflow experiment: {e}")
        raise


def get_databricks_info() -> dict:
    """
    Get Databricks connection information.
    
    Returns:
        dict with host, experiment_name, and tracking_uri
    """
    load_dotenv(Path(__file__).parent.parent / ".env")
    
    return {
        "host": os.getenv("DATABRICKS_HOST"),
        "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME"),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "artifact_root": os.getenv("DATABRICKS_ARTIFACT_ROOT"),
    }


def test_connection() -> bool:
    """
    Test connection to Databricks MLflow.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        setup_mlflow()
        # Try to list experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Successfully connected to Databricks MLflow")
        print(f"   Found {len(experiments)} experiments")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    """Test MLflow Databricks connection"""
    print("🔍 Testing Databricks MLflow connection...")
    print("=" * 50)
    
    # Show configuration
    info = get_databricks_info()
    print("\n📋 Configuration:")
    for key, value in info.items():
        if key != "token" and value:  # Don't print token
            display_value = value if "token" not in key.lower() else "***"
            print(f"   {key}: {display_value}")
    
    print("\n🔌 Testing connection...")
    if test_connection():
        print("\n✅ All good! MLflow is ready to use.")
    else:
        print("\n❌ Connection failed. Check your credentials in .env file.")
