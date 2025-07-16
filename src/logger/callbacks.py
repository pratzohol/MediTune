import wandb
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class WandbLoggingCallback(TrainerCallback):
    """
    Custom callback for logging detailed stats to Weights & Biases.
    """

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        # Add more fields here if needed
        wandb.log(
            {
                "train/loss": logs.get("loss"),
                "train/learning_rate": logs.get("learning_rate"),
                "train/epoch": state.epoch,
            },
            step=state.global_step,
        )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        wandb.log(
            {
                "eval/accuracy": metrics.get("eval_accuracy"),
                "eval/loss": metrics.get("eval_loss"),
                "eval/epoch": state.epoch,
            },
            step=state.global_step,
        )


def build_callbacks():
    return [WandbLoggingCallback()]
