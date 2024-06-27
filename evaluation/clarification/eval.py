import os, argparse, glob, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.data import load_data, extract_results
from common.eval import load_evals, write_evals, fetch_eval_files, Score, open_eval_file
from ast import literal_eval
from tqdm import tqdm
from metrics import rule_based_eval, gpt_evaluation, compute_similarity, fetch_gpt_cache
from filelock import FileLock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--evaluation_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="mistral-7b")
    parser.add_argument("--prompt_method", type=str, default="direct")
    parser.add_argument("--gpt_eval", action="store_true", default=False)
    
    args = parser.parse_args()
    eval_path = os.path.join(args.evaluation_dir, args.data_split)
    eval_files = fetch_eval_files(eval_path)
    final_results_path, final_logs_path = load_evals(eval_path, "ask")
    cleaned_data = load_data(split=args.data_split)
    assert len(eval_files) == len(cleaned_data), "Number of evaluation files does not match the number of data samples"
    
    with FileLock(final_results_path + ".lock"):
        with FileLock(final_logs_path + ".lock"): 
            final_results = open_eval_file(final_results_path, "ask")
            final_logs = open_eval_file(final_logs_path, "ask")
            eval_score = Score()
            for data_idx, eval_data, _ in tqdm(extract_results(eval_files, "ask", args.model_name, args.prompt_method)):
                current_details = literal_eval(cleaned_data[data_idx]['details'])
                for detail_idx, (detail, question) in enumerate(eval_data):
                    if detail is not None:
                        current_details.remove(detail)
                    
                    cover_rate = int((question is None) ^ (len(current_details) > 0))
                    eval_score['clarify_acc', data_idx, detail_idx] = cover_rate
                    eval_score['clear_recall', data_idx, detail_idx] = cover_rate if (not current_details) else None
                    eval_score['vague_recall', data_idx, detail_idx] = cover_rate if current_details else None

                    if len(current_details) > 0 and question is not None:
                        local_scores = {
                            "rule_score": 0,
                            "bleu": 0,
                            "rouge1": 0,
                            "rouge2": 0,
                            "rougeL": 0
                        }
                        if args.gpt_eval:
                            local_scores["gpt_score"] = 0
                        else:
                            gpt_cache = fetch_gpt_cache(final_logs["ask"], args.model_name, args.prompt_method, data_idx, detail_idx)
                            if gpt_cache is not None:
                                local_scores["gpt_score"] = gpt_cache
                        for candidate in current_details:
                            for key, value in compute_similarity(question, candidate).items():
                                local_scores[key] = max(local_scores[key], value)
                            if local_scores["rule_score"] < 1:
                                local_scores["rule_score"] = int(rule_based_eval(question, candidate["attribute"]))
                            if args.gpt_eval and local_scores["gpt_score"] < 1:
                                local_scores["gpt_score"] = int(gpt_evaluation(question, candidate))
                        for key, value in local_scores.items():
                            eval_score[key, data_idx, detail_idx] = value
                            
                assert not current_details, "Not all details are covered"

            final_logs["ask"][f"{args.model_name}_{args.prompt_method}"] = eval_score.get_logs()
            binary_keys = ["clear_recall", "vague_recall", "clarify_acc", "rule_score", "gpt_score"]
            final_results["ask"][f"{args.model_name}_{args.prompt_method}"] = eval_score.calc_scores(macro_binary_keys=binary_keys)
            
            write_evals(eval_path, final_results, final_logs)

        