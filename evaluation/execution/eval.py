import os, argparse, sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common.data import load_data, format_value, extract_results
from common.eval import load_evals, fetch_eval_files, Score, write_evals, open_eval_file
from src.dialogue import generate_trajectories
from metrics import main_eval, well_formed_eval
from filelock import FileLock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--evaluation_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="brute")
    parser.add_argument("--prompt_method", type=str, default="force")
    parser.add_argument("--max_retries", type=int, default=3)

    args = parser.parse_args()
    eval_path = os.path.join(args.evaluation_dir, args.data_split)
    eval_files = fetch_eval_files(eval_path)

    cleaned_data = load_data(split=args.data_split)
    assert len(eval_files) == len(cleaned_data), "Number of evaluation files does not match the number of data samples"
    # pass_rate, api_match, precision, recall, f1
    eval_score = Score()
    if args.prompt_method in ["react", "reflection", "memory"]:
        iteration_data = list(extract_results(eval_files, "tool", args.model_name, args.prompt_method, logs=True))
    else:
        iteration_data = list(extract_results(eval_files, "tool", args.model_name, args.prompt_method))

    for data_idx, result, log in tqdm(iteration_data):
        current = cleaned_data[data_idx]
        for detail_idx, (detail, candidates) in enumerate(result):
            if detail is not None:
                new_value = format_value(detail)
                current[detail["attribute"]] = new_value

            labels = generate_trajectories(current, "finetuning")
            candidates = [candidate for candidate in candidates.split('\n') if candidate]
            main_results = main_eval(candidates, labels)
            if args.prompt_method in ["react", "reflection", "memory"]:
                main_results["steps"] = len(log[detail_idx])
                status_set = [
                    "other error",
                    "tool error",
                    "successful",
                    "finish",
                    "invalid action type",
                    f"retried more than {args.max_retries} times",
                    "null action",
                    f"repeated action more than {args.max_retries} times",
                    "invalid params",
                ]
                for status in status_set:
                    main_results[status] = 0
                for step in log[detail_idx]:
                    main_results[step['status']] = main_results.get(step['status'], 0) + 1
                for status in status_set:
                    main_results[f"{status}_rate"] = main_results.get(status, 0) / main_results["steps"]
            else:
                main_results["well_formed"] = well_formed_eval(candidates)
            for key, value in main_results.items():
                eval_score[key, data_idx, detail_idx] = value

    final_results_path, final_logs_path = load_evals(eval_path, "tool")
    with FileLock(final_results_path + ".lock"):
        with FileLock(final_logs_path + ".lock"):
            final_results = open_eval_file(final_results_path, "tool")
            final_logs = open_eval_file(final_logs_path, "tool")
            final_logs["tool"][f"{args.model_name}_{args.prompt_method}"] = eval_score.get_logs()
            binary_keys = ["pass_rate"]
            final_results["tool"][f"{args.model_name}_{args.prompt_method}"] = eval_score.calc_scores(macro_binary_keys=binary_keys)
            write_evals(eval_path, final_results, final_logs)
