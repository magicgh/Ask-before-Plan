import os, json, glob
from collections import defaultdict

def open_eval_file(output_path: str, task_type: str):
    if not os.path.exists(output_path):
        generated_eval = {}
    else:
        generated_eval = json.load(open(output_path))
        
    if not task_type in generated_eval:
        generated_eval[task_type] = {}
        
    return generated_eval

def fetch_eval_files(evaluation_path: str):
    return list(glob.glob(os.path.join(evaluation_path, "task_*.json")))

def load_evals(evaluation_path: str, task_type: str):
    assert task_type in ["ask", "plan", "tool"]

    result_path, log_path = os.path.join(
        evaluation_path, "evaluation_results.json"
    ), os.path.join(evaluation_path, "evaluation_logs.json")

    return result_path, log_path

def save_eval_file(output_path: str, saved_data):
    with open(output_path, "w") as f:
        json.dump(saved_data, f, indent=4)
    
def write_evals(evaluation_path: str, results, logs):
    save_eval_file(os.path.join(evaluation_path, "evaluation_results.json"), results)
    save_eval_file(os.path.join(evaluation_path, "evaluation_logs.json"), logs)

class Score:
    def __init__(self):
        self.scores = defaultdict(lambda: defaultdict(dict))
    
    def __setitem__(self, key, value: float):
        name, task_idx, turn_idx = key
        self.scores[name][task_idx][turn_idx] = value
        
    def __getitem__(self, key):
        name, task_idx, turn_idx = key
        return self.scores[name][task_idx][turn_idx]
    
    def micro_avg(self, name: str):
        results = [
            value
            for task in self.scores.get(name, {}).values()
            for value in task.values()
            if value is not None
        ]
    
        return sum(results) / len(results) if results else 0
    
    def macro_binary(self, name: str):
        results = []
        for task in self.scores.get(name, {}).values():
            macro_score = 1
            valid_task = 0
            for value in task.values():
                if value is not None:
                    valid_task = 1
                    macro_score = macro_score and value
            if valid_task:
                results.append(macro_score)
        return sum(results) / len(results) if results else 0
    
    def macro_avg(self, name: str):
        results = []
        for task in self.scores.get(name, {}).values():
            local_results = []
            for value in task.values():
                if value is not None:
                    local_results.append(value)
            results.append(sum(local_results) / len(local_results) if local_results else 0)
        return sum(results) / len(results) if results else 0
    
    def get_logs(self):
        return self.scores
    
    def calc_scores(self, macro_binary_keys=[], macro_avg_keys=[]):
        results = {}
        
        assert set(macro_binary_keys).intersection(set(macro_avg_keys)) == set(), "Keys overlap"
        for key in self.scores:
            results[f"micro_{key}"] = self.micro_avg(key)
            if key in macro_binary_keys:
                results[f"macro_{key}"] = self.macro_binary(key)
            elif key in macro_avg_keys:
                results[f"macro_{key}"] = self.macro_avg(key)
                
        return results