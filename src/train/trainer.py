import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        eval_dataset = eval_dataset or self.eval_dataset
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        inputs = [ex["input"] + "\n\nAnswer:" for ex in eval_dataset]

        # Tokenize
        tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=2,
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions = [text[-1] for text in decoded]
        targets = [ex["output"] for ex in eval_dataset]

        accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(targets)
        print(f"\n[MCQA] Generation Accuracy: {accuracy:.4f}\n")

        # calculate eval_loss similar to training loss
        all_input_ids = torch.stack([ex["input_ids"] for ex in eval_dataset]).to(model.device)
        all_attention_mask = torch.stack([ex["attention_mask"] for ex in eval_dataset]).to(model.device)
        all_labels = torch.stack([ex["labels"] for ex in eval_dataset]).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask,
                labels=all_labels,
            )
            loss = outputs.loss

        return {"eval_accuracy": accuracy, "eval_loss": loss.item()}
