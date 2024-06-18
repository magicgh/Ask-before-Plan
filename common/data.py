import os, random, json
from ast import literal_eval
from datasets import load_dataset, Features, Value, Sequence
from copy import deepcopy
from typing import List


def null_transform(sample):

    for column in ["org", "departure_date", "transportation"]:
        sample[column] = sample[column] if sample[column] is not None else ""

    for column in ["dest", "housing", "cuisine"]:
        sample[column] = (
            literal_eval(sample[column]) if sample[column] is not None else []
        )

    for column in ["days", "people_number", "budget"]:
        sample[column] = sample[column] if sample[column] is not None else -1
    return sample


def load_data(file_path: str = 'magicgh/Ask-before-Plan', split: str = "train", start_idx: int = 0, end_idx: int = 1000):

    data_features = Features(
        {
            "task_id": Value("string"),  # Assuming task_id is an integer
            "org": Value("string"),
            "dest": Sequence(Value("string")),
            "days": Value("int32"),
            "departure_date": Value("string"),
            "people_number": Value("int32"),
            "housing": Sequence(
                Value("string")
            ),  # Adjust based on the actual data type
            "cuisine": Sequence(
                Value("string")
            ),  # Adjust based on the actual data type
            "transportation": Value("string"),  # Adjust based on the actual data type
            "budget": Value("int32"),
            "query": Value("string"),
            "level": Value("string"),  # Assuming 'level' is a string
            "details": Value("string"),  # Assuming 'details' is a string
        }
    )

    raw_data = load_dataset(os.path.join(file_path))[split]
    cleaned_data = raw_data.map(null_transform).cast(data_features)
    assert start_idx < end_idx, "start index must be less than end index"
    assert end_idx <= len(cleaned_data), "End index out of range"
    return cleaned_data.select(range(start_idx, end_idx))


def randomize_details(details):
    random.seed(42)
    return sorted(details, key=lambda x: (x["priority"], random.random()))


def format_value(detail):
    if detail["value"] is None:
        if detail["attribute"] in ["housing", "cuisine"]:
            return []
        else:
            return ""
    elif isinstance(detail["value"], float):
        return int(detail["value"])
    else:
        return detail["value"]
        
def expand_task(sample: dict):
    details = randomize_details(literal_eval(sample["details"]))
    current_details = deepcopy(sample)
    current_details.pop("details")
    yield None, deepcopy(current_details)
    for detail in details:
        new_value = format_value(detail)
        current_details[detail["attribute"]] = new_value
        yield detail, deepcopy(current_details)

def precheck_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_results(save_path: str, task_types):
    if isinstance(task_types, str):
        task_types = [task_types]
    assert isinstance(task_types, list)
    for task_type in task_types:
        assert task_type in ["ask", "plan", "tool"]
    
    if not os.path.exists(save_path):
        generated_result = {}
    else:
        generated_result = json.load(open(save_path))
    
    for task_type in task_types:
        if not task_type in generated_result:
            generated_result[task_type] = {}
        
    return generated_result

def write_results(save_path: str, generated_result: dict):
    with open(save_path, "w") as f:
        json.dump(generated_result, f, indent=4)
                
def check_results_integrity(raw_data, task_type: str, extract_key: str):
    assert raw_data, "No evaluation results found"
    assert task_type in raw_data, f"No {task_type} results found"
    assert extract_key in raw_data[task_type], f"No {extract_key} results found"
    return raw_data[task_type][extract_key]

def extract_results(evaluation_files: List[str], task_type: str, model_name: str, prompt_method: str, logs: bool = False):
    assert task_type in ["ask", "plan", "tool"], "Invalid task type"
    for eval_file in evaluation_files:
        with open(eval_file, 'r') as f:
            raw_data = json.load(f)
        eval_data = check_results_integrity(raw_data, task_type, f"{model_name}_{prompt_method}_results")
        if logs:
            log_data = check_results_integrity(raw_data, task_type, f"{model_name}_{prompt_method}_logs")
            yield int(eval_file.split("task_")[-1].split(".json")[0]), eval_data, log_data
        else:
            yield int(eval_file.split("task_")[-1].split(".json")[0]), eval_data, None

def generate_message(role, message):
    if role not in ["user", "assistant"]:
        raise ValueError("Role must be either 'user' or 'assistant'")
    return {role: message}

extract_cities = lambda dest: [schedule.split(" on ")[0].strip() for schedule in dest]
extract_days = lambda dest: [
    int(schedule.split(" on day ")[1].strip()) for schedule in dest
]
