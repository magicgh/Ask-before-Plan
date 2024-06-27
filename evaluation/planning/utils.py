import json
import re
import gradio as gr
import os

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def save_file(data, path):
    with open(path,'w',encoding='utf-8') as w:
        for unit in data:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()

def unique_consecutive(elements):
    if not elements:
        return []

    # Initialize the list with the first element
    result = [elements[0]]

    # Iterate through the list, adding elements that are not the same as the last added element
    for current in elements[1:]:
        if current != result[-1]:
            result.append(current)

    return result

def extract_query_number(query_string):
    """
    Extract the number from a query string formatted as "Query X" or "Query X --- Done".
    
    Args:
    - query_string (str): The input string.
    
    Returns:
    - int: The extracted number if found, else None.
    """
    pattern = r"Query (\d+)"
    match = re.search(pattern, query_string)
    return int(match.group(1)) if match else None

def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"

    
def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

def get_city_list(days, deparure_city, destination):
    city_list = []
    city_list.append(deparure_city)
    if days == 3:
        city_list.append(destination)
    else:
        city_set = open('../database/background/citySet_with_states.txt').read().split('\n')
        state_city_map = {}
        for unit in city_set:
            city, state = unit.split('\t')
            if state not in state_city_map:
                state_city_map[state] = []
            state_city_map[state].append(city)
        for city in state_city_map[destination]:
            if city != deparure_city:
                city_list.append(city + f"({destination})")
    return city_list

def get_filtered_data(component,data, column_name=('NAME','city')):
    name, city = get_valid_name_city(component)
    return data[(data[column_name[0]] == name) & (data[column_name[1]] == city)]

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))  # Add the last group of values
    return result