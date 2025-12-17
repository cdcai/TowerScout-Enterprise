# TowerScout Cloud Adaptation

## Introduction
This repo is an extension of the existing TowerScout application to utilize a more enterprise architecture on Databricks. It's ideal for users of TowerScout who are looking for a more scalable system

## Key Capabilities
This migration introduces the following features:
- Stored detections in ADLS/S3
- Retraining pipeline connected to validated images
- Model scaling and storage in Unity Catalog

## Architecture
![TowerScout Architecture](https://github.com/cdcent/TowerScout/blob/prod/towerscout_architecture.jpg)
## Quickstart
We do not provide the model in this release; however, you can train your own model using the training pipeline.

We developed this using DBR 15.4 ML. Although this application was designed on Databricks, you can adapt the code to a standard Spark setup

The original TowerScout is monolithic and that may serve some use cases better, you can find that repo [here](https://github.com/TowerScout/TowerScout).

### UI
The frontend files are located in webapp

### Backend/ML
The backend/ML files are located in tsdb

#### notebooks
Notebooks are the orchestration tools we used for the PySpark backend. The application runs using file arrival to trigger a structured streaming query. We have a timer listener that closes the stream after a period of time.

## Future Adaptations
We run this using YOLOv5, but if you opt to run this on Databricks, using YOLOv10 along with a ML Endpoint will yield better performance.

Licensed under <a target="_blank" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC-BY-NC-SA-4.0</a>
(see <a target="_blank" href="https://github.com/TowerScout/TowerScout/blob/main/LICENSE.TXT">LICENSE.TXT</a> in the root of the repository for details)
