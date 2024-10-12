import os
import logging
import torch
import argparse
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def train(args):

    assert (
        args.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if "mistral-7b" in args.base_model.lower():
        learning_rate = 2e-5
        lora_r = 64
        lora_alpha = 16
        lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj']
        chat_template = open(os.path.join(current_file_dir, "../configs/chat_templates/mistral-instruct.jinja")).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template
    
    elif "llama-3-8b" in args.base_model.lower():
        learning_rate = 3e-4
        lora_r = 8
        lora_alpha = 16
        lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'gate_proj', 'down_proj', 'up_proj', 'o_proj']
        chat_template = open(os.path.join(current_file_dir, "../configs/chat_templates/llama-3-chat.jinja")).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template
    
    elif "llama-2" in args.base_model.lower():
        learning_rate = 5e-5
        lora_r = 8
        lora_alpha = 16
        lora_target_modules = ['q_proj', 'v_proj']
        chat_template = open(os.path.join(current_file_dir, "../configs/chat_templates/llama-2-chat.jinja")).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template
        
    else:
        raise ValueError(f"Unknown model {args.base_model}")
    
    lora_dropout = 0.05

    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = tokenizer.apply_chat_template(data_point["messages"], tokenize=False, add_generation_prompt=True)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    if args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            args.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logging.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logging.error(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
    val_data = None
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="tensorboard",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)

    logging.info(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="mistral-community/Mistral-7B-v0.2")
    parser.add_argument("--data_path", type=str, default="./sft_data")
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    train(args)
