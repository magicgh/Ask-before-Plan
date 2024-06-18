import os, sys, logging, re, argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ast import literal_eval
from base import Agent, ReactAgent, ReflectAgent
from typing import Dict, List
from langchain.schema import HumanMessage, AIMessage
from common.prompts.tool import tool_prompts, error_prompts, tool_description_set
from common.chat import parse_tool, is_null_action
from execution.utils import action_mapping, params_regex, extract_actions, generate_api_docs
from tools.utils import ToolError
from collections import Counter

from tqdm import tqdm
from langchain_community.callbacks import get_openai_callback
from common.data import load_data, expand_task, load_results, write_results, generate_message, precheck_path
from copy import deepcopy
from filelock import FileLock

logging.basicConfig(level=logging.INFO)


class ToolAgent(Agent):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 1,
        prompt_lib: Dict = tool_prompts["direct"],
        zero_shot: bool = False,
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
        self.zero_shot = zero_shot

    def _build_user_message(self) -> str:
        return [HumanMessage(self._build_user_prompt(conversations=self.conversations))]

    def _build_example_message(self) -> str:
        return [
            HumanMessage(self._build_custom_prompt("example_user")),
            AIMessage(self._build_custom_prompt("example_agent")),
        ]

    def run(self, conversations, reset=True) -> str:

        if reset:
            self.reset()

        self.conversations = conversations

        for step in range(self.max_steps):
            if self.zero_shot:
                self.messages = (
                    self._build_system_message() + self._build_user_message()
                )
            else:
                self.messages = (
                    self._build_system_message()
                    + self._build_example_message()
                    + self._build_user_message()
                )
            results = self.generate()

        return results


class ReactToolAgent(ReactAgent):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 30,
        prompt_lib: Dict = tool_prompts["react"],
        max_retries: int = 3,
        env_names: List[str] = [
            "accommodations",
            "attractions",
            "budget",
            "flights",
            "googleDistanceMatrix",
            "restaurants",
        ],
        port: int = None,
    ):

        super().__init__(
            model_name,
            max_input_tokens,
            max_new_tokens,
            temperature,
            max_steps,
            prompt_lib,
            env_names,
            port=port,
        )
        self.logs = []
        self.current_observation = ""
        self.max_retries = max_retries
        self.retry_count = {key: 0 for key in list(self.envs) + [None, "finish"]}
        self.action_counter = Counter()

    def reset(self) -> None:
        super().reset()
        self.action_counter.clear()
        self.reset_retry_count()
        self.logs.clear()
        self.current_observation = ""

    def reset_retry_count(self) -> None:
        self.retry_count = {key: 0 for key in list(self.envs) + [None, "finish"]}

    def run(self, conversations, reset=True) -> None:

        if reset:
            self.reset()

        self.conversations = conversations

        while not (self.is_halted() or self.is_done()):
            self.step()

        return self.scratchpad, self.logs

    def invoke_agent(self) -> str:
        self.messages = self._build_system_message() + self._build_user_message()
        return self.generate()

    def _build_user_message(self) -> str:
        return [
            HumanMessage(
                self._build_user_prompt(
                    conversations=self.conversations,
                    scratchpad=self.scratchpad,
                )
            )
        ]

    def update_scratchpad(self, content: str = None) -> None:
        if not content:
            content = self.current_observation
        if (
            content
            and self.scratchpad
            and (not content[0].isspace())
            and (not self.scratchpad[-1].isspace())
        ):
            content = " " + content
        self.scratchpad += content

    def handle_invalid_params(self, action_type: str, api_name: str) -> None:
        self.retry_count[api_name] += 1
        self.current_observation = error_prompts["invalid_params"].format(
            action=action_type
        )
        self.update_scratchpad()
        self.logs[-1]["status"] = "invalid params"

    def step(self) -> None:

        self.logs.append(
            {
                "step": self.curr_step,
                "thought": "",
                "action": "",
                "observation": "",
                "status": "",
            }
        )

        # Think
        self.update_scratchpad(f"\nThought {self.curr_step}:")
        self.update_scratchpad(self.invoke_agent())
        logging.info(self.scratchpad.split("\n")[-1])

        self.logs[-1]["thought"] = (
            self.scratchpad.split("\n")[-1]
            .replace(f"Thought {self.curr_step}:", "")
            .strip()
        )

        # Act
        self.update_scratchpad(f"\nAction {self.curr_step}:")
        action = self.invoke_agent()

        if is_null_action(action):
            self.update_scratchpad(error_prompts["null_action"])
        else:
            self.update_scratchpad(action)

        # refresh action_counter
        self.action_counter[action] += 1

        self.logs[-1]["action"] = (
            self.scratchpad.split("\n")[-1]
            .replace(f"Action {self.curr_step}:", "")
            .strip()
        )

        # examine if the same action has been repeated 3 times
        if self.action_counter[action] > self.max_retries:
            logging.warning(
                f"'{action}' is repeated more than {self.max_retries} times. Early stopping."
            )
            self.logs[-1]["status"] = f"repeated action more than {self.max_retries} times"
            self.done = True
            return

        logging.info(self.scratchpad.split("\n")[-1])

        # Observe
        self.update_scratchpad(f"\nObservation {self.curr_step}:")

        if is_null_action(action):
            action_type, params = None, None
            self.update_scratchpad(error_prompts["no_feedback"])
            logging.error(
                f"Observation {self.curr_step}: "
                + "No feedback from the environment due to the null action."
            )
            self.logs[-1][
                "observation"
            ] = "No feedback from the environment due to the null action."
            
            self.logs[-1]["status"] = "null action"

        else:
            action_type, params = parse_tool(action)
            api_name = action_mapping.get(action_type, None)
            if (
                action_type != "Finish"
                and self.retry_count[api_name] + 1 > self.max_retries
            ):
                logging_action_name = (
                    action_type if api_name is not None else "invalidAction"
                )
                logging.warning(
                    f"'{logging_action_name}' is retried more than {self.max_retries} times. Early stopping."
                )
                self.logs[-1]["status"] = f"retried more than {self.max_retries} times"
                self.done = True
                return

            if api_name is None or params is None:
                self.retry_count[api_name] += 1
                self.current_observation = error_prompts["invalid_action"].format(
                    action=action
                )
                self.update_scratchpad()
                self.logs[-1]["status"] = "invalid action type"
                logging.error("Invalid action type detected.")

            elif action_type == "Finish":
                if len(params) > 0:
                    logging.error(
                        f"Error in parsing parameters for {action_type}: Finish should not have any parameters."
                    )
                    self.handle_invalid_params(action_type, api_name)
                else:
                    self.done = True
                    self.reset_retry_count()
                    self.logs[-1]["status"] = "finish"
                    return

            else:
                try:
                    matched_params = re.match(params_regex[action_type], params)
                    eval_params = list(map(literal_eval, matched_params.groups()))

                except Exception as e:
                    logging.error(f"Error in parsing parameters for {action_type}: {e}")
                    self.handle_invalid_params(action_type, api_name)

                else:
                    try:
                        self.current_observation = self.envs[api_name].view(
                            *eval_params
                        )
                        self.update_scratchpad()
                        self.reset_retry_count()
                        self.logs[-1]["status"] = "successful"

                    except ToolError as e:
                        logging.error(e)
                        self.retry_count[api_name] += 1
                        self.current_observation = str(e)
                        self.update_scratchpad()
                        self.logs[-1]["status"] = "tool error"

                    except Exception as e:
                        logging.error(e)
                        self.retry_count[api_name] += 1
                        self.current_observation = (
                            f"Illegal {action_type}. Please try again."
                        )
                        self.update_scratchpad()
                        self.logs[-1]["status"] = "other error"

            logging.info(f"Observation {self.curr_step}: " + self.current_observation)
            self.logs[-1]["observation"] = self.current_observation.strip()

        self.curr_step += 1


class ReflectToolAgent(ReactToolAgent, ReflectAgent):

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 30,
        prompt_lib: Dict = tool_prompts["reflect"],
        max_retries: int = 3,
        env_names: List[str] = [
            "accommodations",
            "attractions",
            "budget",
            "flights",
            "googleDistanceMatrix",
            "restaurants",
        ],
        port: int = None,
    ):
        super().__init__(
            model_name,
            max_input_tokens,
            max_new_tokens,
            temperature,
            max_steps,
            prompt_lib,
            max_retries,
            env_names,
            port=port,
        )
        
    def _build_user_message(self) -> str:
        return [
            HumanMessage(
                self._build_user_prompt(
                    conversations=self.conversations,
                    scratchpad=self.scratchpad,
                    reflections=self.format_rationales(),
                )
            )
        ]

    def last_action_failed(self) -> bool:
        return self.logs[-1]["status"] != "successful" and self.logs[-1]["status"] != "finish"
    
    def run(self, conversations, reset=True, past_memory=None) -> None:

        if reset:
            self.reset()

        if past_memory:
            self.rationales = past_memory
        
        self.conversations = conversations

        while not (self.is_halted() or self.is_done()):
            self.step()
            if self.last_action_failed() and not self.is_done():
                self.reflect()

        return self.scratchpad, self.logs, self.rationales
    
    def _build_reflection_message(self) -> str:
        last_action = self.logs[-1]["action"]
        return [
            HumanMessage(
                self._build_reflection_prompt(
                    tool_docs=generate_api_docs(action=last_action, api_docs=tool_description_set),
                    observation=self.logs[-1]["observation"],
                    action=last_action,
                )
            )
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prompt_method", type=str, default="direct", choices=["direct", "zero_shot", "react", "reflection", "memory"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))

    if args.prompt_method == "direct":
        agent = ToolAgent(
            model_name=args.model_name,
            max_input_tokens=2048,
            max_new_tokens=800,
            temperature=0,
            port=args.port,
        )
    elif args.prompt_method == "zero_shot":
        agent = ToolAgent(
            model_name=args.model_name,
            max_input_tokens=2048,
            max_new_tokens=800,
            temperature=0,
            zero_shot=True,
            port=args.port,
        )
    elif args.prompt_method == "react":
        agent = ReactToolAgent(
            model_name=args.model_name,
            max_input_tokens=30000,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
        )
    elif args.prompt_method == "reflection":
        agent = ReflectToolAgent(
            model_name=args.model_name,
            max_input_tokens=30000,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
        )
    elif args.prompt_method == "memory":
        agent = ReflectToolAgent(
            model_name=args.model_name,
            max_input_tokens=30000,
            max_new_tokens=256,
            temperature=0,
            port=args.port,
        )

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)

    with get_openai_callback() as cb:
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            tool_results, scratchpads, action_logs, rationale_logs = [], [], [], []
            messages = [generate_message("user", sample["query"])]
            for modified, current in expand_task(sample):
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))

                if args.prompt_method in ["direct", "zero_shot"]:
                    tool_results.append((modified, agent.run(messages)))
                    
                elif args.prompt_method in ["react"]:
                    scratchpad, action_log = agent.run(messages)
                    tool_results.append((modified, extract_actions(action_log)))
                    scratchpads.append(scratchpad)
                    action_logs.append(deepcopy(action_log))
                    
                elif args.prompt_method in ["reflection"]:
                    scratchpad, action_log, rationale_log = agent.run(messages)
                    tool_results.append((modified, extract_actions(action_log)))
                    scratchpads.append(scratchpad)
                    action_logs.append(deepcopy(action_log))
                    rationale_logs.append(deepcopy(rationale_log))
                    
                elif args.prompt_method in ["memory"]:
                    previous_memory = None if len(rationale_logs) == 0 else deepcopy(rationale_logs[-1])
                    scratchpad, action_log, rationale_log = agent.run(messages, past_memory=previous_memory)
                    tool_results.append((modified, extract_actions(action_log)))
                    scratchpads.append(scratchpad)
                    action_logs.append(deepcopy(action_log))
                    rationale_logs.append(deepcopy(rationale_log))

            logging.info(f"Tool generated for sample {args.start_idx + data_idx}")

            save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
            with FileLock(save_path + '.lock'):
                generated_result = load_results(save_path, "tool")

                generated_result["tool"][
                    f"{args.model_name}_{args.prompt_method}_results"
                ] = tool_results

                if args.prompt_method in ["react", "reflection", "memory"]:
                    generated_result["tool"][
                        f"{args.model_name}_{args.prompt_method}_logs"
                    ] = action_logs

                    generated_result["tool"][
                        f"{args.model_name}_{args.prompt_method}_scratchpad"
                    ] = scratchpads
                
                if args.prompt_method in ["reflection", "memory"]:
                    generated_result["tool"][
                        f"{args.model_name}_{args.prompt_method}_rationales"
                    ] = rationale_logs

                # write to json file
                write_results(save_path, generated_result)
            logging.info(f"Tool saved for sample {args.start_idx + data_idx}")
        logging.info(cb)
