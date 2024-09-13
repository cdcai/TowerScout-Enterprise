# Databricks notebook source
#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# TowerScout Team:
# Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#

# YOLOv5 detector class

import math
import torch
import sys
import numpy as np
from mlflow.pyfunc import PythonModel, PythonModelContext
from torch import nn


class YOLOv5_Detector(
    PythonModel
):  # needs to follow the Protocol in inference pipeline
    def __init__(self, model, batch_size):
        self.model = model  # model should be passed as a param to init, create model outside of class
        self.batch_size = (
            batch_size  # For our Tesla K8, this means 8 batches can run in parallel
        )

    def predict(
        self,
        context: PythonModelContext,
        model_input: list[np.ndarray],
        secondary: nn.Module = None,
    ):  # change detect to predict to follow mlflow conventions and remove crop
        # Inference in batches
        results = []
        count = 0
        #print(f"Detecting with secondary model {secondary}")

        for i in range(0, len(model_input), self.batch_size):
            # img_batch = model_input[i:i+self.batch_size] #[Image.open(tile['filename']) for tile in tile_batch]
            # the model expectes a LIST of images, list of np arrays or Image objects.
            # a np array of images doesn't appear to work
            # see: ultralytics_yolov5_master/models/common.py
            # error: ValueError: axes don't match array
            img_batch = [model_input[j] for j in range(i, i + self.batch_size)]
            # retain a copy of the images
            if secondary is not None:
                img_batch2 = [img.copy() for img in img_batch]
            else:
                img_batch2 = [None] * len(img_batch)

            # detect, remove self.semaphore and events.query

            result_obj = self.model(img_batch)

            # get the important part
            results_raw = result_obj.xyxyn
            # print(results_raw)
            # result is tile by tile, imma just remove "tile" here
            for img, result in zip(img_batch2, results_raw):
                results_cpu = result.cpu().numpy().tolist()

                # secondary classifier processing
                if secondary is not None:
                    # classifier will append its own prob to every detection
                    secondary.classify(img, results_cpu, batch_id=count)
                    count += 1

                # yolo_resuts(*item) named tuple
                tile_results = [
                    {
                        "x1": item[0],
                        "y1": item[1],
                        "x2": item[2],
                        "y2": item[3],
                        "conf": item[4],
                        "class": int(item[5]),
                        "class_name": result_obj.names[int(item[5])],
                        "secondary": item[6] if len(item) > 6 else 1,
                    }
                    for item in results_cpu
                ]
                results.append(tile_results)

                # record the detections in the tile
                # boxes = []
                # for tr in tile_results:
                #     box = "0 " + \
                #         str((tr['x1']+tr['x2'])/2) + \
                #         " "+str((tr['y1']+tr['y2'])/2) + \
                #         " "+str(tr['x2']-tr['x1']) +\
                #         " "+str(tr['y2']-tr['y1'])+"\n"
                #     boxes.append(box)
                # tile['detections'] = boxes

            #print(f" batch of {len(img_batch)} processed")

        return results
