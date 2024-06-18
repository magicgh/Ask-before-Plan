from typing import List

action_mapping = {
    "AccommodationSearch": "accommodations",
    "RestaurantSearch": "restaurants",
    "AttractionSearch": "attractions",
    "DistanceMatrix": "googleDistanceMatrix",
    "FlightSearch": "flights",
    "BudgetEstimator": "budget",
    "Finish": "finish",
}

quoted_string_pattern = r'("[^"]*"|\'[^\']*\')'
list_pattern = r"(\[[^\]]*\])"
int_pattern = r"(\d+)"

params_regex = {
    "AccommodationSearch": rf"^{quoted_string_pattern},\s*{list_pattern}$",
    "RestaurantSearch": rf"^{quoted_string_pattern},\s*{list_pattern}$",
    "AttractionSearch": rf"^{quoted_string_pattern}$",
    "DistanceMatrix": rf"^{quoted_string_pattern},\s*{quoted_string_pattern},\s*{quoted_string_pattern}$",
    "FlightSearch": rf"^{quoted_string_pattern},\s*{quoted_string_pattern},\s*{quoted_string_pattern}$",
    "BudgetEstimator": rf"^{quoted_string_pattern},\s*{list_pattern},\s*{list_pattern},\s*{int_pattern}$",
}

def extract_actions(action_logs: List[dict]):
    action_list = []
    for action_log in action_logs:
        if action_log.get("status", "") == "successful":
           action_list.append(action_log["action"].strip())
    return "\n".join(action_list)

def generate_api_docs(action: str, api_docs: dict):
    
    api_categories = [
        ("accommodation", "accommodations"),
        ("restaurant", "restaurants"),
        ("attraction", "attractions"),
        ("distance", "googleDistanceMatrix"),
        ("flight", "flights"),
        ("budget", "budget"),
        ("finish", "finish")
    ]
    
    action = action.lower()
    for index, (keyword, category) in enumerate(api_categories):
        if keyword in action:
            return api_docs[category].replace(f"{index + 1}. ", "")
    else:
        return "\n".join(api_docs[cat[1]] for cat in api_categories)
        