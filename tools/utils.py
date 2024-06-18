import os, re, logging
import pandas as pd
from typing import List


city_set_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/background/citySet.txt"))
city_set = []

lines = open(city_set_path, 'r').read().strip().split('\n')
for unit in lines:
    city_set.append(unit)

class ToolError(Exception):
    def __init__(self, text: str):
        self.text = text
    def __str__(self):
        return f'Tool error: {self.text}.'

class DateError(ToolError):
    def __init__(self, date: str):
        self.date = date
    def __str__(self):
        return f'Illegal parameters: "{self.date}" is not in the format of YYYY-MM-DD.'
        

class CityError(ToolError):
    def __init__(self, city: str):
        self.city = city
    def __str__(self):
        return f'Illegal parameters: "{self.city}" is not a valid city.'

class FilterError(ToolError):
    def __init__(self, filter_str: str):
        self.filter_str = filter_str
    def __str__(self):
        return f'Illegal parameters: "{self.filter_str}" is not a valid accommodation filter. Valid filters are "private room", "entire room", "not shared room", "shared room", "smoking", "parties", "children under 10", "visitors", "pets".'

class CuisineError(ToolError):
    def __init__(self, cuisine: str):
        self.cuisine = cuisine
    def __str__(self):
        return f'Illegal parameters: "{self.cuisine}" is not a valid cuisine. Valid cuisines are "Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", and "French".'
    
class TransportationError(ToolError):
    def __init__(self, mode: str):
        self.mode = mode
    def __str__(self):
        return f'Illegal parameters: "{self.mode}" is not a valid transportation mode. Valid modes are "self-driving" and "taxi".'

class PeopleNumberError(ToolError):
    def __init__(self, people_number: int):
        self.people_number = people_number
    def __str__(self):
        return f'Illegal parameters: "{self.people_number}" is not a valid number of people. The number of people should be a positive integer.'

def validate_date_format(dates: List[str]) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    for date_str in dates:
        if not re.match(pattern, date_str):
            raise DateError(date_str)
    return True

def validate_city_format(cities: List[str]) -> bool:
    for city_str in cities:
        if city_str not in city_set:
            raise CityError(city_str)
    return True

def to_string(data, verbose=True) -> str:
    if data is not None:
        if type(data) == pd.DataFrame:
            if verbose:
                return data.to_string(index=False).strip()
            else:
                return data.head(n=3).to_string(index=False).strip() + "\n..."
        else:
            return str(data).strip()
    else:
        return str(None)

def extract_from_to(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B" in the given text, with B ending at a comma or the end of the string.
    
    Args:
    - text (str): The input string.
    
    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        logging.warning(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"