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

#
# EfficientNet B5 secondary classifier
# Looks at detections between x% and y% confidence, rechecks
#

import torch
import torch.nn as nn
import mlflow 

from efficientnet_pytorch import EfficientNet
from torchvision import transforms

import PIL
from PIL import Image
from tsdb.ml.utils import cut_square_detection, YOLOv5Detection


class ENClassifier(nn.Module):

    def __init__(self, model: nn.Module):
        super(ENClassifier, self).__init__()
        """
        If you intend to fine tune this model you may need a 
        proxy connection which may not be available in production.
        """
        self.model = model
        
        # switch to GPU memory if available
        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()

        # prepare the image transform
        self.transform = transforms.Compose([
            transforms.Resize([456, 456]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5553, 0.5080, 0.4960), std=(0.1844, 0.1982, 0.2017))
            ])
    
    @classmethod
    def from_uc_registry(cls, model_name: str, alias: str):
        """
        Create EN_calssifer object using a registered model from UC Model Registry
        """
        registered_model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}@{alias}"
        )

        return cls(registered_model)


    def forward(self, x):
        return x

    #
    # classify:
    #
    # take batch of one img and multiple detections
    # returns filtered detections (class 0 only)
    #
    def classify(self, 
                img: Image, 
                detections: list[YOLOv5Detection],
                min_conf: float = 0.25, 
                max_conf: float = 0.65, 
                batch_id: int = 0
    ) -> None:
        
        count=0
        for det in detections:
            x1,y1,x2,y2,conf = det[0:5]

            # only for certain confidence range
            if conf >= min_conf and conf <= max_conf:
                det_img = cut_square_detection(img, x1, y1, x2, y2)

                # now apply transformations
                input = self.transform(det_img).unsqueeze(0)

                # put on GPU if we have one
                if torch.cuda.is_available():
                    input = input.cuda()

                # and feed into model
                # this is 1-... because the secondary has class 0 as tower
                output = 1 - torch.sigmoid(self.model(input).cpu()).item()
                print(" inspected: YOLOv5 conf:",round(conf,2), end=", ")
                print(" secondary result:", round(output,2))
                #img.save("uploads/img_for_id_"+f"{batch_id+count:02}_conf_"+str(round(conf,2))+"_p2_"+str(round(output,2))+".jpg")
                #det_img.save("uploads/id_"+f"{batch_id+count:02}_conf_"+str(round(conf,2))+"_p2_"+str(round(output,2))+".jpg")
                p2 = output

            elif conf < min_conf:
                print(" No chance: YOLOv5 conf:", round(conf,2))
                # garbage gets thrown out right here
                p2 = 0

            else:
                # >= max_conf does not need review, gets added to results
                print(" kept: YOLOv5 conf:", round(conf,2))
                p2 = 1

            det.append(p2)
            count += 1



