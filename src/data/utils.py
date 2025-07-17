import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


def get_tokenized_datasets(dataset: DatasetDict, tokenizer: PreTrainedTokenizer, max_seq_length: int = 512):
    """
    Applies formatting and tokenization to each split of the dataset.

    Args:
        dataset (DatasetDict): The dataset to tokenize (e.g., with 'train', 'validation' splits).
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text.
        max_seq_length (int, optional): The maximum sequence length after tokenization. Defaults to 512.

    Returns:
        DatasetDict: Tokenized version of the dataset.
    """

    def format_and_tokenize(example):
        # Combine input and answer in causal LM style
        full_text = example["input"] + "\n\nAnswer -->  " + example["output"]

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        # Labels = input_ids (causal LM objective)
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask out padding tokens for loss
        tokenized["labels"] = [
            label if mask == 1 else -100 for label, mask in zip(tokenized["labels"], tokenized["attention_mask"])
        ]

        return tokenized

    print("Tokenizing datasets...")
    tokenized_dataset = DatasetDict(
        {
            split: dataset[split].map(
                format_and_tokenize,
                desc=f"Tokenizing {split} split",
            )
            for split in dataset
        }
    )
    return tokenized_dataset


def load_and_prepare_medmcqa(
    max_examples: int = 5000,
    output_path: str = "dataset/processed",
    split_ratio: float = 0.8,
):
    """
    Loads the MedMCQA dataset from Hugging Face, formats it into input/output pairs,
    creates train/val/test splits, and saves the processed version to disk.

    Args:
        max_examples (int): Maximum number of examples to use for train+val+test.
        output_path (str): Directory where processed dataset will be saved.
        split_ratio (float): Fraction of data to use for training. Remaining is split equally between val and test.
    """

    output_path = Path(output_path)

    # Check if processed folder exists and is non-empty
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Found existing processed dataset at: {output_path}, skipping raw processing.")
        return

    print("Loading MedMCQA dataset from HuggingFace...")
    dataset = load_dataset("openlifescienceai/medmcqa", cache_dir="dataset/raw")

    def format_mcq(example):
        input_text = f"{example['question']}\n\nOptions:\n"

        for i, opt in enumerate(["opa", "opb", "opc", "opd"]):
            input_text += f"\n{chr(65 + i)}. {example[opt]}"

        return {"input": input_text, "output": f"{chr(65 + example['cop'])}"}

    print("Formatting and splitting...")
    ntrain = int(split_ratio * max_examples)
    nval = int((1 - split_ratio) * max_examples / 2)
    ntest = int((1 - split_ratio) * max_examples / 2)

    # MedMCQA's original 'test' split lacks correct answers.
    # So we simulate test/val splits from the training set.
    dataset["test"] = dataset["validation"].select(range(ntest))
    dataset["validation"] = dataset["train"].select(range(ntrain, ntrain + nval))
    dataset["train"] = dataset["train"].select(range(ntrain))

    # Format dataset
    formatted_dataset = DatasetDict(
        {
            split: dataset[split].map(
                format_mcq,
                remove_columns=dataset[split].column_names,
                desc=f"Formatting {split} split",
            )
            for split in dataset
        }
    )

    # creates intermediary parent folders if it doesn't exist
    # exist_ok doesn't raise error if directory is already present
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed dataset to: {output_path}")
    formatted_dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    load_and_prepare_medmcqa(5)
