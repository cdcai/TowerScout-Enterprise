# Databricks notebook source
import mlflow
from tsdb.ml.utils import PromotionArgs, Hyperparameters
from tsdb.ml.data import data_augmentation, DataLoaders
from tsdb.ml.train import model_promotion

# COMMAND ----------

client = mlflow.MlflowClient()


# client.delete_registered_model(name='edav_dev_csels.towerscout.yolo_detection_model')  # for deletion once experimenting are done
# loaded_model = mlflow.pytorch.load_model('runs:/ad3c356c5b444698b95979d7d2fdbf2e/towerscout_model')

# registration_info = mlflow.register_model('runs:/ad3c356c5b444698b95979d7d2fdbf2e/towerscout_model', name="edav_dev_csels.towerscout.yolo_detection_model")
# client.set_registered_model_alias(name="edav_dev_csels.towerscout.yolo_detection_model", alias="prod", version=registration_info.version)
# print(registration_info.version)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")


hyperparameters = Hyperparameters(
    lr0=0.0,
    momentum=0.0,
    weight_decay=0.0,
    batch_size=16,
    epochs=4,
    prob_H_flip=0,
    prob_V_flip=0,
    prob_mosaic=0,
)

cache_dir = "/local/cache/path"
out_root_base = "/Volumes/edav_dev_csels/towerscout/data/mds_training_splits/test_image_gold/version=397"

dataloaders = DataLoaders.from_mds(
    cache_dir,
    mds_dir=out_root_base,
    hyperparams=hyperparameters,
    transforms=None,
)

promo_args = PromotionArgs(
    challenger_uri="runs:/ec50663136d5413b89b1600e6d66c7f2/towerscout_model",
    testing_dataloader=dataloaders.test,
    comparison_metric="f1",
    model_name="edav_dev_csels.towerscout.yolo_detection_model",
    alias="prod",
)

model_promotion(promo_args)
