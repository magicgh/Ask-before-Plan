import sys, os, re, random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from tqdm import tqdm
import argparse
from common.data import load_data, expand_task, load_results, write_results, precheck_path
from common.chat import parse_tool
from dialogue import generate_trajectories
from typing import Dict
from filelock import FileLock


def check_accommodation(text: str):
    return "there is no accommodation in this city" in text.lower()


def check_attraction(text: str):
    return "there is no attraction in this city" in text.lower()


def check_flight(text: str):
    return "there is no flight from" in text.lower()


def check_distance_matrix(text: str):
    return "no valid information" in text.lower()


def check_cuisines(text: str):
    pattern = r"there is no \w+ restaurant in this city"
    return re.search(pattern, text.lower()) is not None


function_unfeasible_mapping = {
    "AccommodationSearch": [
        [
            "Your accommodation preference is not found in the city. Would you like to consider other options?"
        ],
        check_accommodation,
    ],
    "AttractionSearch": [
        ["No attraction found in the city. Do you want to change your destination?"],
        check_attraction,
    ],
    "FlightSearch": [
        [
            "I cannot find any flight. Do you want to change your destination?",
            "I cannot find any flight. Do you want to change your transportation preference?",
        ],
        check_flight,
    ],
    "DistanceMatrix": [
        [
            "No valid transportation information found. Do you want to change your destination?",
            "No valid transportation information found. Do you want to change your transportation preference?",
        ],
        check_distance_matrix,
    ],
    "RestaurantSearch": [
        [
            "Your cuisine preference is not found in the city. Would you like to consider other options?"
        ],
        check_cuisines,
    ],
}

function_missing_mapping = {
    "AccommodationSearch": [
        "Could you provide information about your destinations and your arrival dates?"
    ],
    "AttractionSearch": [
        "Could you provide information about your destinations and your arrival dates?"
    ],
    "FlightSearch": [
        "Could you provide information about your destinations and your arrival dates?",
        "Could you tell me your departure city for this trip?",
        "Could you tell me your departure date for this trip?",
        "How many days would you like to spend on this trip?",
    ],
    "DistanceMatrix": [
        "Could you provide information about your destinations?",
        "Could you tell me your departure city for this trip?",
    ],
    "RestaurantSearch": [
        "Could you provide information about your destinations and your arrival dates?"
    ],
    "BudgetEstimator": [
        "Could you provide information about your destinations and your arrival dates?",
        "Could you tell me your departure city for this trip?",
        "Could you tell me your departure date for this trip?",
        "How many days would you like to spend on this trip?",
        "How many people are on this trip?",
        "What is your budget for this trip?",
    ]
}


def generate_unfeasible_questions(function_name: str, observation: str):

    if function_unfeasible_mapping.get(function_name) is not None:
        if function_unfeasible_mapping[function_name][1](observation):
            return random.choice(function_unfeasible_mapping[function_name][0])

    return None


def select_question(trajectories: Dict):
    question_candidates = []
    function_set = set()
    for function, observation in trajectories.items():
        function_name = parse_tool(function)[0]
        question = generate_unfeasible_questions(function_name, observation)
        if question is not None:
            question_candidates.append(question)
        function_set.add(function_name)
    
    for function in function_missing_mapping.keys():
        if function not in function_set:
            question_candidates.append(random.choice(function_missing_mapping[function]))
    
    if len(question_candidates) > 0:
        return random.choice(question_candidates)
    else:
        return None


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
        ask_results = []
        for modified, current in expand_task(sample):
            trajectories = generate_trajectories(current, output_format="clarification")
            ask_results.append((modified, select_question(trajectories)))

        save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
        with FileLock(save_path + '.lock'):
            generated_result = load_results(save_path, "ask")
            generated_result["ask"]["environment_only_results"] = ask_results
            write_results(save_path, generated_result)
