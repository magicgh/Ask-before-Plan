import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from planning.utils import parse_plan, check_subplan_format, ParseError
import logging, argparse
from ast import literal_eval
from typing import Dict, List
from base import Agent, ReactAgent, ReflectAgent
from common.prompts.plan import planner_prompts, error_prompts
from langchain.schema import HumanMessage
from common.data import load_data, write_results, expand_task, load_results, generate_message, precheck_path
from common.chat import parse_json_tool, is_null_action
from collections import Counter

from copy import deepcopy
from dialogue import generate_trajectories
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
from filelock import FileLock

logging.basicConfig(level=logging.INFO)


class Planner(Agent):

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 1,
        prompt_lib: Dict = planner_prompts["direct"],
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

    def run(self, conversations, trajectories, reset=True) -> str:

        if reset:
            self.reset()

        self.conversations = conversations
        self.trajectories = trajectories
        results = None
        for step in range(self.max_steps):
            self.messages = self._build_system_message() + self._build_user_message()
            if self._exceed_max_tokens():
                logging.error("Exceeding max tokens.")
                break
            results = self.generate()

        try:
            return parse_plan(results)
        except Exception as e:
            logging.error(e)
            return None


class ReactPlanner(ReactAgent):
    """
    A question answering ReAct Agent.
    """

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 30,
        prompt_lib: Dict = planner_prompts["react"],
        max_retries: int = 3,
        env_names: List[str] = ["calculator"],
        port: int = None,
    ) -> None:

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
        self.envs = self.envs[env_names[0]]
        self.results = None
        self.action_counter = Counter()
        self.max_retries = max_retries

    def _build_user_message(self) -> str:
        return [
            HumanMessage(
                self._build_user_prompt(
                    conversations=self.conversations,
                    trajectories=self.trajectories,
                    scratchpad=self.scratchpad,
                )
            )
        ]
    
    def _build_instruction(self) -> str:
        return self._build_system_message() + self._build_user_message()
    
    def invoke_agent(self) -> str:
        self.messages = self._build_instruction()
        return self.generate()
        
    def run(self, conversations, trajectories, reset=True) -> None:

        if reset:
            self.reset()

        self.conversations = conversations
        self.trajectories = trajectories

        while not (self.is_halted() or self.is_done()):
            self.step()

        return self.results, self.scratchpad, self.logs

    def update_scratchpad(self, content: str = None) -> None:
        if (
            content
            and self.scratchpad
            and (not content[0].isspace())
            and (not self.scratchpad[-1].isspace())
        ):
            content = " " + content
        self.scratchpad += content
        
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
        self.update_scratchpad(action)
        self.action_counter[action] += 1
        
        self.logs[-1]["action"] = (
            self.scratchpad.split("\n")[-1]
            .replace(f"Action {self.curr_step}:", "")
            .strip()
        )
        
        if self.action_counter[action] > self.max_retries:
            logging.warning(
                f"'{action}' is repeated for more than {self.max_retries} times. Early stopping."
            )
            self.logs[-1]["status"] = f"repeated action more than {self.max_retries} times"
            self.done = True
            return
        
        logging.info(self.scratchpad.split("\n")[-1])

        # Observe
        self.update_scratchpad(f"\nObservation {self.curr_step}: ")

        action_type, params = parse_json_tool(action)

        if action_type == "CostEnquiry":
            try:
                input_arg = literal_eval(params)
                if not isinstance(input_arg, dict):
                    raise ValueError(
                        error_prompts["invalid_subplan"]
                    )
                if check_subplan_format(input_arg):
                    observation = f"Cost: {self.envs.run(input_arg)}"
                    self.logs[-1]["status"] = "cost enquired"
            
            except ParseError as e:
                observation = str(e)
                self.logs[-1]["status"] = "parse error"
                logging.error(e)
                
            except SyntaxError:
                observation = (
                    error_prompts["error_subplan"]
                )
                self.logs[-1]["status"] = "syntax error"
                logging.error("Syntax error in subplan")
                
            except ValueError as e:
                observation = str(e)
                self.logs[-1]["status"] = "value error"
                logging.error(e)
            
            except Exception as e:
                observation = str(e)
                self.logs[-1]["status"] = "other error"
                logging.error(e)

        elif action_type == "Finish":
            try:
                self.results = parse_plan(params)
                observation = f"The plan is finished."
                self.done = True
                self.logs[-1]["status"] = "finish"
            except ParseError as e:
                observation = str(e)
                self.logs[-1]["status"] = "parse error"
                logging.error(e)
            except Exception as e:
                observation = str(e)
                self.logs[-1]["status"] = "other error"
                logging.error(e)

        elif is_null_action(action):
            observation = error_prompts["null_action"]
            logging.error("No feedback from the environment due to the null action.")
            self.logs[-1]["status"] = "null action"
            
        else:
            observation = error_prompts["invalid_action"]
            logging.error(f"Invalid action: {action}")
            self.logs[-1]["status"] = "invalid action"

        self.curr_step += 1

        self.logs[-1]["observation"] = observation.strip()
        self.update_scratchpad(observation)
        logging.info(self.scratchpad.split("\n")[-1])

    def _exceed_max_tokens(self, custom_messages: List[str] = None) -> bool:
        if custom_messages:
            input_tokens = "\n".join([message.content for message in custom_messages])
        else:
            input_tokens = "\n".join([message.content for message in self._build_instruction()])
        return self._get_token_length(input_tokens) > self.max_input_tokens

    def reset(self) -> None:
        super().reset()
        self.results = None
        self.action_counter.clear()
        self.logs.clear()

class ReflectPlanner(ReactPlanner, ReflectAgent):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float = 0,
        max_steps: int = 30,
        prompt_lib: Dict = planner_prompts["reflect"],
        max_retries: int = 3,
        env_names: List[str] = ["calculator"],
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
                    trajectories=self.trajectories,
                    scratchpad=self.scratchpad,
                    reflections=self.format_rationales(),
                )
            )
        ]


    def run(self, conversations, trajectories, reset=True) -> None:

        if reset:
            self.reset()

        self.conversations = conversations
        self.trajectories = trajectories

        while not (self.is_halted() or self.is_done()):
            self.step()
            if self.envs.is_terminated() and not self.is_done():
                self.reflect()

        return self.results, self.scratchpad, self.logs


    def _build_reflection_message(self) -> str:
        return [
            HumanMessage(
                self._build_reflection_prompt(
                    conversations=self.conversations,
                    trajectories=self.trajectories,
                    scratchpad=self.scratchpad,
                )
            )
        ]

    def reset(self) -> None:
        super().reset()
        self.envs.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prompt_method", type=str, default="react", choices=["direct", "cot", "react", "reflection"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))

    if args.prompt_method == "direct":
        agent = Planner(
            model_name=args.model_name,
            max_input_tokens=16000,
            max_new_tokens=2000,
            temperature=0,
            prompt_lib=planner_prompts["direct"],
            port=args.port,
        )
    elif args.prompt_method == "cot":
        agent = Planner(
            model_name=args.model_name,
            max_input_tokens=16000,
            max_new_tokens=3000,
            temperature=0,
            prompt_lib=planner_prompts["cot"],
            port=args.port,
        )
    
    elif args.prompt_method == "react":
        agent = ReactPlanner(
            model_name=args.model_name,
            max_input_tokens=18000,
            max_new_tokens=2000,
            temperature=0,
            prompt_lib=planner_prompts["react"],
            port=args.port,
        )
    elif args.prompt_method == "reflection":
        agent = ReflectPlanner(
            model_name=args.model_name,
            max_input_tokens=18000,
            max_new_tokens=2000,
            temperature=0,
            prompt_lib=planner_prompts["reflect"],
            port=args.port,
        )

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)
    
    with get_openai_callback() as cb:
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            messages = [generate_message("user", sample["query"])]
            expanded_tasks = list(expand_task(sample))
            for modified, current in expanded_tasks:
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))
            trajectories = generate_trajectories(expanded_tasks[-1][-1], output_format="planning")
            
            if args.prompt_method in ["react", "reflection"]:
                planner_results, scratchpad, planner_logs = agent.run(messages, trajectories)
                logging.info(planner_results)
            else:
                planner_results = agent.run(messages, trajectories)
                logging.info(planner_results)
            
            logging.info(f"Plan generated for sample {args.start_idx + data_idx}")

            save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
            with FileLock(save_path + '.lock'):
                generated_result = load_results(save_path, "plan")
                    
                if args.prompt_method in ["react", "reflection"]:
                    generated_result["plan"][f'{args.model_name}_{args.prompt_method}_scratchpad'] = scratchpad
                    generated_result["plan"][f'{args.model_name}_{args.prompt_method}_logs'] = deepcopy(planner_logs)
                    
                generated_result["plan"][f"{args.model_name}_{args.prompt_method}_results"] = deepcopy(planner_results)
                
                
                # write to json file
                write_results(save_path, generated_result)
            logging.info(f"Plan saved for sample {args.start_idx + data_idx}")
        logging.info(cb)
