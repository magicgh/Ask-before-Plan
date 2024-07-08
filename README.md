<h1 align="center">
<em>Ask-before-Plan</em> <br>
Proactive Language Agents for Real-World Planning
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2406.12639">Paper</a> •
  <a href="https://huggingface.co/datasets/magicgh/Ask-before-Plan">Data</a> •
  <a href="https://drive.google.com/file/d/1vMIhs8mpMgk33pFDv2rWg6AJNyD70Sod">Environment</a> •
  Checkpoints
</p>

## Installation
To install this repository, follow the steps below:
1. Clone the repository:
```bash
git clone https://github.com/magicgh/Ask-before-Plan
```
2. Install the dependencies:
```bash
cd Ask-before-Plan
pip install -r requirements.txt
```
3. Download the `environment.zip` and extract it into the `Ask-before-Plan` directory.

## Model Setup
To use OpenAI models, `OPENAI_API_KEY` must be set in the environment.
```bash
export OPENAI_API_KEY=<API_KEY>
```

For open-source models, we utilize vllm to deploy them as OpenAI compatible servers.  
```bash
# LLaMA-3-8B
python3 -m vllm.entrypoints.openai.api_server \
--served-model-name "llama-3-8b-instruct" \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--kv-cache-dtype fp8 \
--port 10086 \
--chat-template ./configs/chat_templates/llama-3-chat.jinja
```

```bash
# Mistral-7B
python3 -m vllm.entrypoints.openai.api_server \
--served-model-name "mistral-7b-instruct" \
--model mistralai/Mistral-7B-Instruct-v0.2 \
--kv-cache-dtype fp8 \
--port 10087 \
--chat-template ./configs/chat_templates/mistral-instruct.jinja
```  

## Trajectory Tuning
Please ensure the `trajectory_tuning` folder is available in your dataset directory. 
Otherwise, clone the dataset from the [Hugging Face Hub](https://huggingface.co/datasets/magicgh/Ask-before-Plan) into the `data` directory.
You can also use the [conversion script](./preprocess/prepare_finetune_data.py) to convert the raw data into the supervised finetuning format.
### Clarification
Use the following script to finetune the LLaMA-3-8B model:
```bash
#!/bin/bash
python3 src/finetune.py \
    --data_path ./data/trajectory_tunning/clarification_train.json \
    --output_dir ./models/llama-3-8b-ask \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 8 \
    --micro_batch_size 1 
```
Use the following script to finetune the Mistral-7B model:
```bash
#!/bin/bash
python3 src/finetune.py \
    --data_path ./data/trajectory_tunning/clarification_train.json \
    --output_dir ./models/mistral-7b-ask \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --batch_size 8 \
    --micro_batch_size 1 
```
### Execution
Use the following script to finetune the LLaMA-3-8B model:
```bash
#!/bin/bash
python3 src/finetune.py \
    --data_path ./data/trajectory_tunning/execution_train.json \
    --output_dir ./models/llama-3-8b-tool \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 16 \
    --micro_batch_size 2 \
    --warmup_steps 50 
```
Use the following script to finetune the Mistral-7B model:
```bash
#!/bin/bash
python3 src/finetune.py \
    --data_path ./data/trajectory_tunning/execution_train.json \
    --output_dir ./models/mistral-7b-tool \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --batch_size 16 \
    --micro_batch_size 2 \
    --warmup_steps 50 
```
## Agent Inference
Here are some common arguments used in agent inference scripts:
```bash
--data_split # Dataset split to use, e.g., train, test
--output_dir # Directory to save the results, e.g., ./output
--start_idx # Start index of the data samples, e.g., 0
--end_idx # End index of the data samples, e.g., 1000
--port # Port to connect to the model server, e.g., 10086.
```
Note that the `--port` argument is unavailable for Environment-only, Brute-force, Greedy Search, and the Integral Framework.

### Clarification
* Environment-only: `python3 src/clarification/environment_only.py [--data_split ...]`
* Conversation-only 
  * Proactive: `export PROMPT_METHOD=proactive`
  * ProCoT: `export PROMPT_METHOD=procot`

  ```bash
  python3 src/clarification/conversation_only.py \
  --model_name gpt-3.5-turbo \
  --prompt_method $PROMPT_METHOD \
  [--data_split ...]
  ```
* Environment + Conversation
  * Direct (GPT-3.5): `export MODEL_NAME=gpt-3.5-turbo`
  * CEP (Mistral-7B): `export MODEL_NAME=mistral-7b-ask`
  * CEP (LLaMA-3-8B): `export MODEL_NAME=llama-3-8b-ask`
  ```bash
  python3 src/clarification/consultation.py \
  --model_name $MODEL_NAME \
  --prompt_method direct \
  [--data_split ...]
  ```

### Execution
* Brute-force: `python3 src/execution/brute_force.py`
* LLM-based Agent
  * Static Setting
    * Direct (GPT-3.5)
      ```bash
      export MODEL_NAME=gpt-3.5-turbo
      export PROMPT_METHOD=direct
      ```
    * CEP (Mistral-7B)
      ```bash
      export MODEL_NAME=mistral-7b-tool
      export PROMPT_METHOD=zero_shot
      ```
    * CEP (LLaMA-3-8B)
      ```bash
      export MODEL_NAME=llama-3-8b-tool
      export PROMPT_METHOD=zero_shot
      ```
  * Dynamic Setting
    * ReAct and Reflexion
      ```bash
      export MODEL_NAME=gpt-3.5-turbo # or mistral-7b-instruct
      export PROMPT_METHOD=react # or reflection
      ```
    * CEP
      ```bash
      export MODEL_NAME=gpt-3.5-turbo # or mistral-7b-instruct
      export PROMPT_METHOD=memory
      ```
  ```bash
  python3 src/execution/navigator.py \
  --model_name $MODEL_NAME \
  --prompt_method $PROMPT_METHOD \
  [--data_split ...]
  ```

### Planning
* Greedy Search: `python3 src/planning/greedy_search.py`
* LLM-based Agent
  ```bash
  export MODEL_NAME=gpt-3.5-turbo or mistral-7b-instruct
  export PROMPT_METHOD=direct # or cot, react, reflection
  python3 src/planning/planner.py \
  --model_name $MODEL_NAME \
  --prompt_method $PROMPT_METHOD \
  [--data_split ...]
  ```
### Integral Framework
```bash
export BASE_MODEL=llama-3-8b
export PLANNER_MODEL=gpt-3.5-turbo
export PROMPT_METHOD=all
export ASK_PORT=10086
export TOOL_PORT=10086

python3 src/run_integral.py \
--base_model_name $BASE_MODEL \
--planner_model_name $PLANNER_MODEL \
--prompt_method $PROMPT_METHOD \
--ask_port $ASK_PORT \
--tool_port $TOOL_PORT \
[--data_split ...]
```

## Evaluation
Common arguments used in evaluation scripts:
```bash
--data_split # Dataset split to use, e.g., train, test
--evaluation_dir # Directory of the inference results, e.g., ./output
--model_name # Model name to evaluate, e.g., gpt-3.5-turbo
--prompt_method # Prompt method to evaluate, e.g., direct
```
### Clarification
To enable the GPT-4 evaluation, add the `--gpt_eval` flag.
```bash
python3 evaluation/clarification/eval.py \
--gpt_eval \
[--data_split ...]
```
### Execution
```bash
python3 evaluation/execution/eval.py \
[--data_split ...]
```
### Planning
```bash
python3 evaluation/planning/eval.py \
[--data_split ...]
```

## Release Checklist
* Code
  - [x] Baselines
  - [x] Trajectory Tuning Scripts
  - [x] CEP Framework
  - [x] Evaluation Scripts
* Data
  - [x] Ask-before-Plan Dataset
  - [x] Ask-before-Plan Environment
  - [x] Trajectory Tuning Dataset
  - [ ] Trajectory Tuning Checkpoints

## License

Our code is licensed under the [MIT License](./LICENSE).  
The Ask-before-plan dataset and environment are available under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

## Citation
If you find our research helpful for your work, please [![GitHub Repo stars](https://img.shields.io/github/stars/magicgh/ask-before-plan?style=social)](https://github.com/magicgh/Ask-before-Plan) this repository and cite our paper:
```
@article{ask-before-plan,
    author = {Xuan Zhang and Yang Deng and Zifeng Ren and See-Kiong Ng and Tat-Seng Chua},
    journal = {ArXiv preprint},
    title = {Ask-before-Plan: Proactive Language Agents for Real-World Planning},
    url = {https://arxiv.org/abs/2406.12639},
    year = {2024}
}
```
