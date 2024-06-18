import json, re
from typing import List
from ast import literal_eval
class ParseError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self) -> str:
        return self.message

def extract_json(text: str) -> str:
    pattern = r"\[\s*\{[^{}]*\}(?:\s*,\s*\{[^{}]*\})*\s*\]"
    match = re.search(pattern, text)
    if match is None:
        raise ParseError("No valid JSON plan found; please adhere to the example format")
    return match.group()

    
def check_json_format(json_data: str) -> bool:
    keys_in_string = ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
    if not isinstance(json_data, list):
        raise ParseError("JSON plan must be a list")
    if len(json_data) < 2:
        raise ParseError("JSON plan must have at least two days")
    for item in json_data:
        if not isinstance(item, dict):
            raise ParseError("JSON plan must be a list of dictionaries")
        if "current_city" not in item:
            raise ParseError('key "current_city" is missing in the plan')
        
        for key in keys_in_string:
            if key not in item:
                continue
            if not isinstance(item[key], str):
                raise ParseError(f'Value of key "{key}" must be a string')
        
    return True

def parse_plan(plan: str) -> List[dict]:
    
    json_text = extract_json(plan)
    try:
        json_data = json.loads(json_text)
    
    except json.JSONDecodeError as json_error:
        try:
            json_data = literal_eval(json_text)
        
        except Exception as eval_error:
            raise ParseError(f"Failed to parse using JSON decoder: {json_error}\nFailed to parse using eval: {eval_error}")
        
    if check_json_format(json_data):
        return json_data
    
def check_subplan_format(subplan: dict):
    if 'people_number' not in subplan:
        raise ParseError('Key "people_number" is missing in the subplan')
    if not isinstance(subplan['people_number'], int):
        raise ParseError('Value of key "people_number" must be an integer')
    
    keys_in_string = ["transportation", "breakfast", "lunch", "dinner", "accommodation"]
    
    for key in keys_in_string:
        if key not in subplan:
            continue
        if not isinstance(subplan[key], str):
            raise ParseError(f'Value of key "{key}" must be a string')
        
    return True
        
    