from transformers.integrations import WandbCallback
import pandas as pd
import numpy as np


def decode_predictions(tokenizer, predictions):
# <<<<<<< HEAD
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     labels = tokenizer.batch_decode(predictions.label_ids)
#     logits = predictions.predictions.argmax(axis=-1)
#     prediction_text = tokenizer.batch_decode(logits)
# ||||||| parent of 48e3bbe (feat)
#     labels = tokenizer.batch_decode(predictions.label_ids)
#     logits = predictions.predictions.argmax(axis=-1)
#     prediction_text = tokenizer.batch_decode(logits)
# =======
    pd = tokenizer.pad_token_id
    label_ids = np.where(predictions.label_ids != -100, predictions.label_ids, pd)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1), skip_special_tokens=True)
    return {"labels": labels, "predictions": prediction_text}



class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # generate predictions
        predictions = self.trainer.predict(self.sample_dataset)
        # decode predictions and labels
        predictions = decode_predictions(self.tokenizer, predictions)
        # add predictions to a wandb.Table
        predictions_df = pd.DataFrame(predictions)
        predictions_df["epoch"] = state.epoch
        predictions_df["global_step"] = state.global_step
        records_table = self._wandb.Table(dataframe=predictions_df)
        # log the table to wandb
        self._wandb.log({"sample_predictions": records_table})
