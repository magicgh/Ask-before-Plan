import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
                
from tqdm import tqdm
import argparse
from common.data import load_data, expand_task, load_results, write_results, precheck_path
from dialogue import generate_trajectories
from filelock import FileLock

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))
    

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)

    for data_idx, sample in enumerate(tqdm(cleaned_data)):
        tool_results = []
        for modified, current in expand_task(sample):
            tool_results.append((modified, "\n".join(generate_trajectories(current, output_format="brute_force"))))
        
        save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
        with FileLock(save_path + '.lock'):
            generated_result = load_results(save_path, "tool")
            generated_result["tool"]["brute_force_results"] = tool_results
            write_results(save_path, generated_result)
