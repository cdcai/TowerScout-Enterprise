import os
import torch
import torch.distributed as dist
import sys
import torch.nn as nn
from ultralytics import YOLO, settings  # Assuming you're using YOLO from Ultralytics
import mlflow


def calculate_classification_accuracy(precision, recall):
    """Calculate classification accuracy from precision and recall."""
    if precision + recall == 0:
        return 0.0
    accuracy = (precision * recall) / (precision + recall - (precision * recall))
    return accuracy

def demo_yolo8_ddp():
    
    # Manually set distributed environment variables
    os.environ['RANK'] = os.getenv('RANK', '0')  # Default to 0 if not set
    os.environ['WORLD_SIZE'] = os.getenv('WORLD_SIZE', '1')  # Default to 1 if not set
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '127.0.0.1')  # Default to localhost
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')  # Default port for distributed

    # Initialize distributed environment
    dist.init_process_group(backend="nccl", init_method='env://')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get device ID and set up DDP
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    print(f"Training on rank {rank}/{world_size}, device {device_id}")

    # Define training parameters
    train_args = {
        "batch_size": int(sys.argv[1]),
        "epochs": int(sys.argv[2]),
        "exp_name": sys.argv[3],
        "objective_metric": "accuracy",
        "metrics": ["accuracy"]
    }

    # Path to your data.yaml file
    data_yaml = '/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data/data.yaml'

    # using nu model version because it is better apparently 
    yolo_model = YOLO(sys.argv[4])

    # Train the model using your dataset
    yolo_model.train(
        data=data_yaml,  # Path to your dataset YAML file
        epochs=train_args["epochs"],  # Number of training epochs
        imgsz=640,  # Image size
        batch=train_args["batch_size"],  # Batch size
        name=train_args["exp_name"],  # Name of the model
        device=device_id,  # Use GPU device assigned by DDP
        workers=2,  # Number of workers
    )

    # After training, validate the model on the validation set
    metrics = yolo_model.val()

    # The metrics are stored in results_dict, directly access them
    results = metrics.results_dict

    # Extract relevant metrics
    precision = results['metrics/precision(B)']
    recall = results['metrics/recall(B)']
    map50 = results['metrics/mAP50(B)']
    map50_95 = results['metrics/mAP50-95(B)']

    # Calculate classification accuracy
    classification_accuracy = calculate_classification_accuracy(precision, recall)

    # Print out the extracted metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP@50: {map50}")
    print(f"mAP@50-95: {map50_95}")
    print(f"Classification Accuracy: {classification_accuracy}")

    # Test the model using the test data
    test_metrics = yolo_model.val(split='test')  # Use the test split for evaluation

    # Extract and print the relevant metrics for test data
    test_results = test_metrics.results_dict
    test_precision = test_results['metrics/precision(B)']
    test_recall = test_results['metrics/recall(B)']
    test_map50 = test_results['metrics/mAP50(B)']
    test_map50_95 = test_results['metrics/mAP50-95(B)']
    test_classification_accuracy = calculate_classification_accuracy(test_precision, test_recall)

    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test mAP@50: {test_map50}")
    print(f"Test mAP@50-95: {test_map50_95}")
    print(f"Test Classification Accuracy: {test_classification_accuracy}")

    # Clean up DDP
    dist.destroy_process_group()

if __name__ == "__main__":
    # Update a setting
    # Disable MLflow tracking by setting the environment variable
    settings.update({"mlflow": False})
    demo_yolo8_ddp()



