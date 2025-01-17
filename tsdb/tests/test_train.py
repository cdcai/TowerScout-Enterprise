"""
This module tests code in tsdb.ml.train, specifically the perform_pass() and model_promotion() functions.
If needed, function docstrings can include examples of what is being tested.
"""
import pytest
from pyspark.sql import SparkSession
from unittest.mock import MagicMock, patch
from tsdb.ml.train import model_promotion
from tsdb.ml.utils import PromotionArgs
import mlflow
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def promo_args_mock():
    """
    Fixture for creating a PromotionArgs object for testing.
    The logger,client, and test_conv are mocked
    """
    mock_logger = MagicMock()
    mock_client = MagicMock()

    mock_test_conv = MagicMock()
    mock_test_conv.__len__.return_value = 10

    promo_args = PromotionArgs(
        model_name="test_model",
        alias="TEST",
        metrics=["F1"],
        objective_metric="F1",
        model_version=2,
        batch_size=1,
        challenger_metric_value=0.9,
        logger=mock_logger,
        client=mock_client,
        test_dataloader=mock_test_conv
    )

    return promo_args

@pytest.fixture
def sample_context_args() -> dict:
    """
    Fixture for creating a context_args dict for testing.
    batch_size determines the number of steps per epoch
    """
    return {
        "batch_size": 2
    }

@pytest.fixture
def sample_report_interval() -> int:
    """
    Fixture for creating a report_interval int for testing.
    """
    return 2

@pytest.fixture
def sample_epoch_num() -> int:
    """
    Fixture for creating a epoch_num int for testing.
    """
    return 1

@pytest.fixture
def mock_step_func():
    """
    Fixture for creating a step_func function for testing 
    The return value is hardcoded
    """
    step_func = MagicMock(return_value={"F1_TEST":0.8})
    return step_func
 
@pytest.fixture
def mock_converter():
    """
    Fixture for creating a converter object with a set length for testing.
    """
    converter = MagicMock()
    converter.__len__.return_value = 10
    return converter

# def test_model_promotion_challenger_better(promo_args_mock):
#     """
#     Tests that the challenger model is promoted when the 
#     challenger model has a better metric than the champion model.
#     Uses the tsdb.ml.train.model_promotion function.
#     """
#     promo_args = promo_args_mock
#     promo_args.challenger_metric_value = 0.9

#     champion_score = 0.85 # champ worse than challenger
#     perform_pass_result = {
#         f"{promo_args.objective_metric}_TEST": champion_score
#     }

#     fake_model = MagicMock()

#     with patch("mlflow.pytorch.load_model", return_value=fake_model) as mock_load_model, \
#         patch("tsdb.ml.train.perform_pass", return_value=perform_pass_result):
        
#         model_promotion(promo_args)

#         # verify that champ model was loaded
#         mock_load_model.assert_called_once_with(
#             model_uri=f"models:/{promo_args.model_name}@{promo_args.alias}"
#         )

#         # challenger model better ==> it should be promoted
#         promo_args.client.set_registered_model_alias.assert_called_once_with(
#             name=promo_args.model_name,
#             alias=promo_args.alias,
#             version=promo_args.model_version
#         )
#         promo_args.logger.info.assert_any_call(
#             f"Promoting challenger model to {promo_args.alias}."
#         )


# def test_model_promotion_champion_better(promo_args_mock):
#     """
#     Test that the challenger model is not promoted when the 
#     challenger model has a worse metric than the champion model.
#     Uses the tsdb.ml.train.model_promotion function.
#     """
#     promo_args = promo_args_mock
#     promo_args.challenger_metric_value = 0.9

#     champion_score = 0.95 # champ worse than challenger
#     perform_pass_result = {
#         f"{promo_args.objective_metric}_TEST": champion_score
#     }

#     fake_model = MagicMock()

#     with patch("mlflow.pytorch.load_model", return_value=fake_model) as mock_load_model, \
#         patch("tsdb.ml.train.perform_pass", return_value=perform_pass_result):
        
#         model_promotion(promo_args)

#         # verify that champ model was loaded
#         mock_load_model.assert_called_once_with(
#             model_uri=f"models:/{promo_args.model_name}@{promo_args.alias}"
#         )

#         # champ model better ==> no promotion
#         promo_args.client.set_registered_model_alias.assert_not_called()
#         promo_args.logger.info.assert_any_call(
#             f"Challenger model does not perform better than current {promo_args.alias} model. Promotion aborted."
#         )