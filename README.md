# Astro Sales Forecasting MLOps Platform

## Overview

A production-ready MLOps platform for sales forecasting that demonstrates modern machine learning engineering practices. Built on Astronomer (Apache Airflow), this project implements an end-to-end ML pipeline with ensemble modeling, comprehensive visualization, and real-time inference capabilities via Streamlit.

### 🚀 Key Features

* **Automated ML Pipeline**: End-to-end orchestration with Astronomer/Airflow
* **Ensemble Modeling**: Combines XGBoost, LightGBM, and Prophet for robust predictions
* **Advanced Visualizations**: Comprehensive model performance analysis and comparison
* **Real-time Inference**: Streamlit-based web UI for interactive predictions
* **Experiment Tracking**: MLflow integration for model versioning and metrics
* **Distributed Storage**: MinIO S3-compatible object storage for artifacts
* **Containerized Deployment**: Docker-based architecture for consistency

## 🏗️ Architecture

### Technology Stack

<table style="border: 1px solid #d3d3d3; border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid #d3d3d3; padding: 8px;">Component</th>
    <th style="border: 1px solid #d3d3d3; padding: 8px;">Technology</th>
    <th style="border: 1px solid #d3d3d3; padding: 8px;">Purpose</th>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>Orchestration</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Astronomer (Airflow 3.0+)</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Workflow automation and scheduling</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>ML Tracking</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">MLflow 2.9+</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Experiment tracking and model registry</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>Storage</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">MinIO</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">S3-compatible artifact storage</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>ML Models</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">XGBoost, LightGBM, Prophet</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Ensemble forecasting</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>Visualization</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Matplotlib, Seaborn, Plotly</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Model analysis and insights</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>Inference UI</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Streamlit</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Interactive prediction interface</td>
  </tr>
  <tr>
    <td style="border: 1px solid #d3d3d3; padding: 8px;"><strong>Containerization</strong></td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Docker & Docker Compose</td>
    <td style="border: 1px solid #d3d3d3; padding: 8px;">Environment consistency</td>
  </tr>
</table>

## 🚀 Quick Start
### Prerequisites
* Docker Desktop installed and running
* Astronomer CLI (brew install astro on macOS, other OS, you can follow the instructions here)
* 8GB+ RAM available for Docker
* Ports 8080, 8501, 5001, 9000, 9001 available