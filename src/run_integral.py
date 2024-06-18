import os, logging
from dialogue import proceed_action
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
from common.data import load_data, expand_task, load_results, write_results, generate_message, precheck_path
import argparse
from clarification.consultant import AskAgent
from execution.navigator import ToolAgent
from planning.planner import Planner
from filelock import FileLock
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--base_model_name", type=str, default="llama-3-8b")
    parser.add_argument("--planner_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--prompt_method", type=str, default="base", choices=["base", "clarify", "all"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--ask_port", type=int, default=20010)
    parser.add_argument("--tool_port", type=int, default=20020)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))


    ask_agent = AskAgent(
        model_name=args.base_model_name + "-ask",
        max_input_tokens=4096,
        max_new_tokens=256,
        temperature=0,
        port=args.ask_port
    )
    
    tool_agent = ToolAgent(
            model_name=args.base_model_name + "-tool",
            max_input_tokens=2048,
            max_new_tokens=800,
            temperature=0,
            zero_shot=True,
            port=args.tool_port,
        ) 
    
    plan_agent = Planner(
            model_name=args.planner_model_name,
            max_input_tokens=16000,
            max_new_tokens=2000,
            temperature=0,
        )

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)
    
    with get_openai_callback() as cb:
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            ask_results, tool_results = [], []
            messages = [generate_message("user", sample["query"])]
            iteration_data = expand_task(sample)
            if args.prompt_method == "base":
                iteration_data = [list(iteration_data)[0]]
            trajectories = {}
            for modified, _ in iteration_data:
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))
                    
                trajectories = tool_agent.run(messages)
                tool_results.append((modified, trajectories))
                if args.prompt_method == "clarify":
                    ask_results.append((modified, ask_agent.run(messages, proceed_action(trajectories, output_format='clarification'))))
                if args.prompt_method == "all":
                    question = ask_agent.run(messages, proceed_action(trajectories, output_format='clarification'))
                    if question is None:
                        break
                    
            planner_results = plan_agent.run(messages, proceed_action(trajectories, output_format='planning'))
            logging.info(f"Results generated for sample {args.start_idx + data_idx}")

            save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
            with FileLock(save_path + '.lock'):
                generated_result = load_results(save_path, ["ask", "tool", "plan"])
                generated_result["ask"][f"{args.base_model_name + '-ask'}_{args.prompt_method}_results"] = ask_results
                generated_result["tool"][f"{args.base_model_name + '-tool'}_{args.prompt_method}_results"] = tool_results
                generated_result["plan"][f"{args.planner_model_name}_{args.prompt_method}_results"] = deepcopy(planner_results)
                write_results(save_path, generated_result)
                
            logging.info(f"Results saved for sample {args.start_idx + data_idx}")
        logging.info(cb)
