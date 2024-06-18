import re


def parse_action(string):
    pattern = r"^(\w+)\[([^\]]+)\]$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except:
        return None, None

def parse_tool(string):
    pattern = r"^(\w+)\(([^\)]*)\)$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except:
        return None, None

def parse_json_tool(string):
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except:
        return None, None
    
def is_null_action(action: str) -> bool:
    return action is None or len(action.strip()) == 0

def extract_binary_answer(text: str) -> bool:
    # Extract the first yes or no from the clarification need
    text = text.lower()
    index_yes = text.find('yes')
    index_no = text.find('no')

    # Determine which appears first using the indices
    if index_yes == -1 and index_no == -1:
        return False
    elif index_yes != -1 and (index_no == -1 or index_yes < index_no):
        return True
    else:
        return False