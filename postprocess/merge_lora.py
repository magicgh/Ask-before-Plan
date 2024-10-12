from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse

def merge_lora_to_base_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, args.adapter_name_or_path, torch_dtype=torch.float16)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)
    torch.save(model.state_dict(), f"{args.save_path}/pytorch_model.bin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--adapter_name_or_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    merge_lora_to_base_model(args)