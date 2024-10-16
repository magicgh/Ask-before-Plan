import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from base import Agent
from typing import Dict
from langchain.schema import HumanMessage, AIMessage
from common.prompts.ask import ask_prompts
from common.chat import extract_binary_answer as extract_answer
import logging 
from filelock import FileLock

logging.basicConfig(level=logging.INFO)
    
class AskAgent(Agent):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 1,
        prompt_lib: Dict = ask_prompts["direct"],
        port: int = None,
    ):
        super().__init__(
            model_name,
            max_input_tokens,
            max_new_tokens,
            temperature,
            max_steps,
            prompt_lib,
            port=port,
        )

    def _build_user_message(self) -> str:
        return [
            HumanMessage(
                self._build_user_prompt(
                    conversations=self.conversations, trajectories=self.trajectories
                )
            )
        ]

    def _build_ask_message(self) -> str:
        return [
            HumanMessage(
                self._build_custom_prompt("ask")
            )
        ]
    def _build_agent_message(self) -> str:
        return [
            AIMessage("Yes.")
        ]
        
    def run(self, conversations, trajectories, reset=True) -> str:

        if reset:
            self.reset()

        self.conversations = conversations
        self.trajectories = trajectories        
        question = None
        for step in range(self.max_steps):
            self.messages = self._build_system_message() + self._build_user_message()
            answer = self.generate()
            clarification_need = extract_answer(answer)
            if clarification_need:
                self.messages += self._build_agent_message() + self._build_ask_message()
                question = self.generate()
        
        return question 

class ConversationOnlyAgent(AskAgent):
    def _build_user_message(self) -> str:
        return [
            HumanMessage(
                self._build_user_prompt(
                    conversations=self.conversations
                )
            )
        ]

class FewShotAgent(AskAgent):
    def _build_user_example(self) -> str:
        return [
            HumanMessage(self._build_custom_prompt("neg_example_user")),
            AIMessage(self._build_custom_prompt("neg_example_agent")),
            HumanMessage(self._build_custom_prompt("pos_example_user")),
            AIMessage(self._build_custom_prompt("pos_example_agent")),
        ]
    
    def _build_ask_example(self) -> str:
        return [
            HumanMessage(self._build_custom_prompt("ask_example_user")),
            AIMessage(self._build_custom_prompt("ask_example_agent")),
        ]
        
    def run(self, conversations, trajectories, reset=True) -> str:

        if reset:
            self.reset()

        self.conversations = conversations
        self.trajectories = trajectories        
        question = None
        for step in range(self.max_steps):
            self.messages = self._build_system_message() + self._build_user_example() + self._build_user_message()
            answer = self.generate()
            clarification_need = extract_answer(answer)
            if clarification_need:
                self.messages += self._build_agent_message() + self._build_ask_example() + self._build_ask_message()
                question = self.generate()
        
        return question 
    
import os
from dialogue import generate_trajectories
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
from common.data import load_data, expand_task, load_results, write_results, generate_message, precheck_path
import json, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prompt_method", type=str, default="direct", choices=["direct", "few_shot", "conversation_only"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))

    if args.prompt_method == "direct":
        agent = AskAgent(
            model_name=args.model_name,
            max_input_tokens=4096,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
        )
    elif args.prompt_method == "conversation_only":
        agent = ConversationOnlyAgent(
            model_name=args.model_name,
            max_input_tokens=4096,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
            prompt_lib=ask_prompts["conversation_only"],
        )
    elif args.prompt_method == "few_shot":
        agent = FewShotAgent(
            model_name=args.model_name,
            max_input_tokens=8192,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
            prompt_lib=ask_prompts["direct"],
        )

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)
    
    with get_openai_callback() as cb:
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            ask_results = []
            messages = [generate_message("user", sample["query"])]
            for modified, current in expand_task(sample):
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))
                
                if args.prompt_method == "conversation_only":
                    trajectories = None
                else:
                    trajectories = generate_trajectories(current, output_format="clarification")
                ask_results.append((modified, agent.run(messages, trajectories)))
            
            logging.info(f"Clarification generated for sample {args.start_idx + data_idx}")

            save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
            with FileLock(save_path + '.lock'):
                generated_result = load_results(save_path, "ask")
                generated_result["ask"][f"{args.model_name}_{args.prompt_method}_results"] = ask_results
                write_results(save_path, generated_result)
                
            logging.info(f"Plan saved for sample {args.start_idx + data_idx}")
        logging.info(cb)
