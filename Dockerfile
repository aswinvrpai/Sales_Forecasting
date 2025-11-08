FROM astrocrpublic.azurecr.io/runtime:3.0-12

# Install additional dependencies for ML libraries
# Initially starting with user 'root' to install system packages
USER root
RUN apt-get update && \
    apt-get install -y build-essential git gcc g++ libssl-dev libffi-dev python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to astro user to install Python packages
USER astro

# Environment variables for MLflow and S3 (MinIO) configuration
ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1
