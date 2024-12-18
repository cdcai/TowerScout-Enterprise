# Databricks notebook source
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.tune(use_ray=True, iterations=3)
print(results)
