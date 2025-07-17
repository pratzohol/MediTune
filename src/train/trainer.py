import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, default_data_collator


def custom_collate_fn(batch):
    input_texts = [example["input"] for example in batch]
    output_texts = [example["output"] for example in batch]

    # Tensor fields (assuming all are same length)
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)

    return {
        "input": input_texts,
        "output": output_texts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        self.cfg = kwargs.pop("cfg", None)
        super().__init__(**kwargs)

    def evaluate(self, eval_dataset=None, **kwargs):
        eval_dataset = eval_dataset or self.eval_dataset
        model = self.model
        tokenizer = self.tokenizer
        model.eval()

        batch_size = self.args.per_device_eval_batch_size
        dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = [ex + "\n\nAnswer -->  " for ex in batch["input"]]
            targets = batch["output"]

            tokenized = tokenizer(
                inputs,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_seq_length,
            )
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_new_tokens=1,
                )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions = [text[-1] for text in decoded]

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    labels=batch["labels"].to(model.device),
                )
                total_loss += outputs.loss.item()
                num_batches += 1

        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        avg_loss = total_loss / num_batches

        print(f"\n[MCQA] Generation Accuracy: {accuracy:.4f}")
        return {"eval_accuracy": accuracy, "eval_loss": avg_loss}
