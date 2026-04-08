# ML Workflow for Scones Unlimited

This project demonstrates an end-to-end machine learning workflow for **Scones Unlimited**, a fictional bakery. The workflow automates the process of data ingestion, model inference, and result processing using **Amazon SageMaker**, **AWS Lambda**, and **AWS Step Functions**.

---

## 📁 Project Overview

The workflow includes:

1. **Data Ingestion & Preprocessing** – Handled in SageMaker notebooks and Lambda functions.
2. **Model Inference** – Classifies or predicts using a trained ML model.
3. **Automation & Orchestration** – AWS Step Functions orchestrate the Lambda functions and SageMaker endpoints.
4. **Output Generation** – Stores processed results for analysis.

---

## 🛠️ Project Files

| File/Folder | Description |
|------------|-------------|
| `starter.ipynb` | Jupyter notebook to explore and preprocess data, and interact with the workflow. |
| `lambda_classify.py` | Lambda function to classify or score incoming data. |
| `lambda_filter.py` | Lambda function to filter data based on predefined rules. |
| `lambda_serialize.py` | Lambda function to serialize results for downstream processing. |
| `SconesUnlimitedWorkflow.asl.json` | AWS Step Functions workflow definition in JSON format. |
| `Execution Input_Output.png` | Example input and output of the workflow execution. |
| `stepfunctions_graph.png` | Visual representation of the Step Functions workflow. |

---

## 🚀 How It Works

1. **Trigger Workflow** – The Step Functions state machine starts the workflow.
2. **Data Processing** – Lambda functions preprocess and filter incoming data.
3. **Model Scoring** – SageMaker endpoint scores the data using the trained ML model.
4. **Result Serialization** – Results are serialized and stored for further analysis.
5. **Monitoring** – Workflow execution and metrics can be tracked using AWS CloudWatch.

---

## 🛠️ Tools & Technologies

- **AWS SageMaker** – Model training, deployment, and hosting endpoints.  
- **AWS Lambda** – Serverless functions for data processing and orchestration.  
- **AWS Step Functions** – Orchestrates Lambda functions and SageMaker endpoints.  
- **Python** – Data processing and workflow scripts.  

---
