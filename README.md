<h1 align="center">
<em>Ask-before-Plan</em> <br>
Proactive Language Agents for Real-World Planning
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2406.12639">Paper</a> •
  <a href="https://huggingface.co/datasets/magicgh/Ask-before-Plan">Data</a> •
  Environment •
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
3. Download the environment and extract into the `Ask-before-Plan` directory.


## Model Setup
To use OpenAI models, `OPENAI_API_KEY` must be set in the environment.
```bash
export OPENAI_API_KEY=<API_KEY>
```

For open-source models, we utilize vllm to deploy them as OpenAI compatible servers.  
```bash
# LLaMA-3-8B
python3 -m vllm.entrypoints.openai.api_server --served-model-name "llama-3-8b-instruct" --model meta-llama/Meta-Llama-3-8B-Instruct --kv-cache-dtype fp8 --port 10086 --chat-template ./configs/chat_templates/llama-3-chat.jinja
```

```bash
# Mistral-7B
python3 -m vllm.entrypoints.openai.api_server --served-model-name "mistral-7b-instruct" --model mistralai/Mistral-7B-Instruct-v0.2 --kv-cache-dtype fp8 --port 10087 --chat-template ./configs/chat_templates/mistral-instruct.jinja
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
* Environment-only: `python3 src/clarification/environment_only.py`
* Conversation-only 
  * Proactive: `export PROMPT_METHOD=proactive`
  * ProCoT: `export PROMPT_METHOD=procot`

  ```bash
  python3 src/clarification/conversation_only.py --model_name gpt-3.5-turbo --prompt_method $PROMPT_METHOD
  ```
* Environment + Conversation
  * Direct (GPT-3.5): `export MODEL_NAME=gpt-3.5-turbo`
  * CEP (Mistral-7B): `export MODEL_NAME=mistral-7b-ask`
  * CEP (LLaMA-3-8B): `export MODEL_NAME=llama-3-8b-ask`
  ```bash
  python3 src/clarification/consultation.py --model_name $MODEL_NAME --prompt_method direct
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
  python3 src/execution/navigator.py --model_name $MODEL_NAME --prompt_method $PROMPT_METHOD
  ```

### Planning
* Greedy Search: `python3 src/planning/greedy_search.py`
* LLM-based Agent
  ```bash
  export MODEL_NAME=gpt-3.5-turbo or mistral-7b-instruct
  export PROMPT_METHOD=direct # or cot, react, reflection
  python3 src/planning/planner.py --model_name $MODEL_NAME --prompt_method $PROMPT_METHOD
  ```
### Integral Framework
```bash
export BASE_MODEL=llama-3-8b
export PLANNER_MODEL=gpt-3.5-turbo
export PROMPT_METHOD=all
export ASK_PORT=10086
export TOOL_PORT=10086

python3 src/run_integral.py --base_model_name $BASE_MODEL --planner_model_name $PLANNER_MODEL --prompt_method $PROMPT_METHOD --ask_port $ASK_PORT --tool_port $TOOL_PORT
```

## Release Checklist
* Code
  - [x] Baselines
  - [ ] Trajectory Tuning Scripts
  - [x] CEP Framework
  - [ ] Evaluation Scripts
* Data
  - [x] Ask-before-Plan Dataset
  - [ ] Ask-before-Plan Environment
  - [ ] Trajectory Tuning Dataset
  - [ ] Trajectory Tuning Checkpoints

## Citation
```
@article{ask-before-plan,
    author = {Xuan Zhang and Yang Deng and Zifeng Ren and See-Kiong Ng and Tat-Seng Chua},
    journal = {ArXiv preprint},
    title = {Ask-before-Plan: Proactive Language Agents for Real-World Planning},
    url = {https://arxiv.org/abs/2406.12639},
    year = {2024}
}
```
