from typing import Any
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch import Tensor
from torch.utils.data import DataLoader

import optuna
import mlflow
from mlflow.models.signature import infer_signature, ModelSignature
import ultralytics.utils as uutils

from tsdb.ml.efficientnet import EN_Classifier
from tsdb.ml.utils import Steps, Hyperparameters
from tsdb.ml.data import DataLoaders


@dataclass
class TrainingArgs:
    """
    A class to represent model training arguements

    Attributes:
        objective_metric: The evaluation metric we want to optimize
        report_interval: Interval to log metrics during training
        val_interval: Interval to evaluate the model
        metrics: Various model evaluation metrics we want to track
    """

    objective_metric: str = "BCE"  # will be selected option for the drop down
    report_interval: int = 5
    val_interval: int = 1
    metrics: Any = None
    nbs: float
    momentum: float
    warmup_momentum: float
    warmup_bias_lr: float
    lrf
    cosine_lr: float


class BaseTrainer:
    """
    This base class implements our base trainer for
    training various deep learning models. We mimic the PyTorch Lightning
    interface to allow for easy extension and modification.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        train_args: TrainingArgs = None,
        epochs: int = 1,
        accumulate: int = 3,
        batch_size: int 4,
        lf: float = None,
        scheduler: optim.lr_scheduler.LambdaLR = None
    ):  # pragma: no cover
        # Model
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.args = self.model.args
        self.amp = self.args.amp

        # Hyperparams and Args
        self.train_args = train_args
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulate = accumulate
        
        # Gradients
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp)
            if uutils.torch_utils.TORCH_2_4
            else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        
        # Optimization
        self.lf = None
        self.scheduler = None

    @classmethod
    def from_optuna_hyperparameters(
        cls,
        hyperparameters: Hyperparameters,
        model: nn.Module,
        train_args: TrainingArgs,
    ) -> "BaseTrainer":
        """
        Class method to create a YoloModelTrainer class instance from the Hyperparameters dataclass
        """
        optimizer = cls.build_optimizer(
            model=model,
            name="Adam",  # TODO: can be tuned by Optuna but is not right now
            lr=hyperparameters.lr0,
            momentum=hyperparameters.momentum,
            decay=hyperparameters.weight_decay,
        )

        return cls(
            model,
            optimizer,
            train_args,
            epochs=hyperparameters.epochs,
            batch_size=hyperparameters.batch_size
        )

    def _setup_scheduler(self):
        """
        Initialize training learning rate scheduler. Taken from ultralytics
        """
        lrf = self.train_args.lrf
        
        if self.train_args.cos_lr:
            self.lf = one_cycle(1, lrf, self.epochs)
        else:
            # Linear
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - lrf) + lrf
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)

    @staticmethod
    def build_optimizer(
        model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        """
        Code adapted from the build_optimizer method in: ultralytics/models/yolo/train.py
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.

        # TODO: test this
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            )
            model.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)

        return optimizer

    def optimizer_step(self):  # pragma: no cover
        """
        Perform a single step of the training optimizer with gradient clipping and EMA update.
        Note: We didn't include the self.ema conditional from the source code
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def training_step(self, minibatch) -> dict:
        """
        TODO: test this
        """
        raise NotImplementedError("Child class needs to implement training_step")

    @torch.no_grad()
    def validation_step(
        self, minibatch, step: Steps = Steps.VAL
    ) -> dict:  # pragma: no cover
        raise NotImplementedError("Child class needs to implement validation_step")

    def get_signature(self, dataloader: DataLoader) -> ModelSignature:
        """
        Returns the mlflow signature of the model for logging and registration
        """
        self.model.eval()
        minibatch = next(iter(dataloader))

        # Note: we are using the first image in the batch
        signature = infer_signature(
            model_input=minibatch[0].cpu().numpy(),
            model_output=self.model(minibatch)[0].detach().cpu().numpy(),
        )

        return signature

    @staticmethod
    def perform_pass(
        step_func: callable,
        dataloader: DataLoader,
        report_interval: int,
        epoch_num: int = 0,
    ) -> dict[str, float]:
        """
        Performs a single pass (epoch) over the data accessed by the dataloader

        Args:
            step_func: A callable that performs either a training or inference step
            dataloader: The torch dataloader
            report_interval: How often to report metrics during pass
            epoch_num: The current epoch number for logging metrics across epochs
        Returns:
            dict[str, float] A dict containing values of various metrics for the epoch
        """

        metrics = {}
        num_batches = len(dataloader)

        for minibatch_num, minibatch in enumerate(dataloader):
            metrics = step_func(minibatch=minibatch)

            if minibatch_num % self.train_args.report_interval == 0:
                step_num = minibatch_num + (epoch_num * num_batches)
                mlflow.log_metrics(metrics, step=step_num)

        return metrics

    def manual_warmup(self, num_warmup: int, step: int):
        """
        A manual warmup phase for the optimizer. This is taken from the source code of Ultralytics.
        """
        if step > num_warmup:
            return
        
        x_i = [0, num_warmup]
        self.accumulate = max(1, int(np.interp(step, x_i, [1, self.train_args.nbs / self.batch_size]).round()))

        for index, param in enumerate(self.optimizer.param_groups):
            param["lr"] = np.interp(
                step, 
                x_i, 
                [self.train_args.warmup_bias_lr if index == 0 else 0.0, param["initial_lr"] * self.lf(self.epoch)]
            )

            if "momentum" in param:
                param["momentum"] = np.interp(
                    step,
                    x_i,
                    [self.train_args.warmup_momentum, self.train_args.momentum]
                )

    def train(
        self, 
        dataloaders: DataLoaders, 
        model_name: str = "towerscout_model", 
        trial: optuna.trial.Trial = None
    ) -> dict[str, Any]:  # pragma: no cover
        """
        Trains a model with given hyperparameter values and returns the value
        of the objective metric on the valdiation dataset.

        Args:
            model_trainer: The model trainer
            dataloaders: The dataloaders for the train/val/test datasets
            model_name: The name to log the model under in MLflow
        Returns:
            dict[str, float] A dict containing the loss
        """
        num_batches = len(dataloaders.train)
        num_warmup = (
            max(round(self.train_args.warmup_epochs * num_batches), 100) 
            if self.train_args.warmup_epochs > 0 else -1
        )

        self.optimizer.zero_grad()

        for epoch in range(self.epochs):
            last_optimizer_step = -1
            self.model.train()

            # Manual Warmup
            self.manual_warmup()

            # Train 
            for batch_index, train_batch in enumerate(dataloaders.train):
                step_number = batch_index + (num_batches * epoch)
                metrics = self.training_step(minibatch=train_batch)
                loss = metrics.pop("loss")

                self.scaler.scale(loss).backward()
                metrics["loss_TRAIN"] = loss.cpu().item()

                if step_number - last_optimizer_step >= self.accumulate:
                    self.optimizer_step()
                    last_otpimizer_step = step_number

                if batch_index % self.train_args.report_interval == 0:
                    mlflow.log_metrics(metrics, step=step_number)
            
            # Validation
            if epoch % self.train_args.val_interval == 0:
                self.model.eval()
                val_report_interval = len(dataloaders.val)
                for batch_index, val_batch in enumerate(dataloaders.val):
                    val_metrics = self.validation_step(minibatch=val_batch, step=Steps.VAL)
                    val_metrics["loss_VAL"] = loss.cpu().item()

                    if batch_index % val_report_interval == 0:
                        step_number = batch_index + (num_batches * epoch)
                        mlflow.log_metrics(val_metrics, step=step_number)
                
                val_metric = val_metrics[f"{self.train_args.objective_metric}_VAL"] 
                
                if trial is not None:
                    trial.report(val_metric, step_number)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

        # Done Training
        signature = self.get_signature(dataloaders.val)

        mlflow.pytorch.log_model(
            self.model,
            model_name,
            signature=signature,
        )

        return val_metric

    def save_model(self) -> None:  # pragma: no cover
        pass


def set_optimizer(model, optlr=0.0001, optmomentum=0.9, optweight_decay=1e-4):  # pragma: no cover
    params_to_update = []
    for name, param in model.named_parameters():
        if "_bn" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.SGD(
        params_to_update, lr=optlr, momentum=optmomentum, weight_decay=optweight_decay
    )
    return optimizer


def score(logits, labels, step: str, metrics):  # pragma: no cover
    return {
        f"{metric.name}_{step}": metric.value(logits, labels).cpu().item()
        for metric in metrics
    }
    

@torch.no_grad()
def inference_step(minibatch, model, metrics, step) -> dict:  # pragma: no cover
    model.eval()
    logits, _, labels = forward_func(model, minibatch)
    return score(logits, labels, step, metrics)
