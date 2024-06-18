import os, sys, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from base import Agent
from typing import Dict
from langchain.schema import HumanMessage
from common.prompts.ask import ask_prompts
import logging

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
from common.data import load_data, expand_task, load_results, write_results, generate_message, precheck_path
from filelock import FileLock

logging.basicConfig(level=logging.INFO)


def extract_answer(text: str) -> bool:

    question_prompt = "The clarifying question is"

    if question_prompt in text:
        return text.split(question_prompt)[1].strip()

    elif question_prompt.lower() in text:
        return text.split(question_prompt.lower())[1].strip()

    elif "no action should be taken" in text.lower():
        return None

    else:
        return text


class ConversationAgent(Agent):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 1,
        prompt_lib: Dict = ask_prompts["proactive"],
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
        return [HumanMessage(self._build_user_prompt(conversations=self.conversations))]

    def run(self, conversations, reset=True) -> str:

        if reset:
            self.reset()

        self.conversations = conversations
        question = None
        for step in range(self.max_steps):
            self.messages = self._build_system_message() + self._build_user_message()
            question = extract_answer(self.generate())
        logging.info(question)

        return question


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--prompt_method", type=str, default="proactive", choices=["proactive", "procot"]
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))

    agent = ConversationAgent(
        model_name=args.model_name,
        max_input_tokens=4096,
        max_new_tokens=512,
        temperature=0,
        prompt_lib=ask_prompts[args.prompt_method],
        port=args.port,
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

                ask_results.append((modified, agent.run(messages)))

            logging.info(f"Clarification generated for sample {args.start_idx + data_idx}")

            save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
            with FileLock(save_path + '.lock'):
                generated_result = load_results(save_path, "ask")

                generated_result["ask"][
                    f"{args.model_name}_{args.prompt_method}_results"
                ] = ask_results

                # write to json file
                write_results(save_path, generated_result)
            logging.info(f"Plan saved for sample {args.start_idx + data_idx}")
        logging.info(cb)
