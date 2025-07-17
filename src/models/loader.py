import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(cfg):
    model_cfg = cfg.model
    model_name = model_cfg.pretrained_model_name_or_path
    use_peft = model_cfg.use_peft
    load_in_4bit = model_cfg.load_in_4bit

    print(f"Loading model: {model_name}")

    if use_peft:
        print(f"PEFT: {use_peft} | Type: {'QLoRa' if load_in_4bit else 'LoRa'}")
    else:
        print(f"PEFT: False")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional QLoRA quant config
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
        )

    # Load model (with or without quant config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # we are doing this in TrainingArguments Class in main.py ⬇️
    # model.gradient_checkpointing_enable()

    # Apply PEFT if enabled
    if use_peft:
        # Use this to reduce memory overhead
        # LoRA + QLoRA + gradient_checkpointing is the golden trio for low-VRAM training.

        # Required only for QLoRA: prepares model for 4-bit training
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # By default, wraps {q_proj, v_proj}
        lora_cfg = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    return tokenizer, model
