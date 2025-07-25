import os
import sys
from pathlib import Path

pardir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(pardir)

import hydra
import torch
import wandb
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data.utils import get_tokenized_datasets
from src.logger.callbacks import build_callbacks
from src.models.loader import load_model_and_tokenizer
from src.train.trainer import CustomTrainer


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Load tokenizer + model
    tokenizer, model = load_model_and_tokenizer(cfg)

    # Load processed MedMCQA dataset
    dataset = load_from_disk(cfg.dataset_path)
    tokenized = get_tokenized_datasets(dataset, tokenizer, cfg.max_seq_length)

    # Setup W&B project/user/run
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.run_name,
        config=dict(cfg),
    )

    # Training setup
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=["wandb"],
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=cfg.logging_steps,
        label_names=["labels"],
    )

    # Trainer setup
    trainer = CustomTrainer(
        cfg=cfg,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=build_callbacks(),
    )

    # Launch training
    trainer.train()

    # Save the best model
    output_path = Path(cfg.output_dir) / "best_model"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving best model to: {output_path}")
    model.save_pretrained(output_path)

    # Final evaluation
    print("Evaluation on test set for FINE_TUNED MODEL:")
    test_results = trainer.evaluate(tokenized["test"])
    print(test_results)

    wandb.finish()


if __name__ == "__main__":
    main()
