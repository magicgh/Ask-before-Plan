import os, sys, json, datetime, logging, re
from typing import List, Dict
from ast import literal_eval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))

from common.data import extract_cities, extract_days
from common.chat import parse_tool
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.flights.apis import Flights
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from tools.budget.apis import Budget
from src.execution.utils import params_regex



tool_templates = {
    "accommodation": 'AccommodationSearch("{city}", {filters})',
    "restaurant": 'RestaurantSearch("{city}", {preferences})',
    "attraction": 'AttractionSearch("{city}")',
    "distance_matrix": 'DistanceMatrix("{origin}", "{destination}", "{mode}")',
    "flight": 'FlightSearch("{origin}", "{destination}", "{date}")',
    "budget": 'BudgetEstimator("{org}", {dest}, {dates}, {people_number})'
}

tool_descriptions = {
    "accommodation": "Accommodations in {city} with filters: {filters}",
    "restaurant": "List all restaurants in {city} and check the availability of cuisines: {preferences}",
    "attraction": "Attractions in {city}",
    "distance_matrix": "{mode} from {origin} to {destination}",
    "flight": "Flight from {origin} to {destination} on {date}",
    "budget": "Estimated minimum budget for {people_number} people from {org} to {dest} for each departure on {dates}"
}

accommodation = Accommodations()
restaurant = Restaurants()
flight = Flights()
google_distance_matrix = GoogleDistanceMatrix()
attraction = Attractions()
budget = Budget(accommodation, flight, google_distance_matrix)

def camel_to_snake(text):
    # Add an underscore before each capital letter and convert the whole string to lowercase
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    return snake_case

def proceed_action(action_str: str, output_format: str):
    assert output_format in ['planning', 'clarification'], "Invalid output format"
    trajectories = {}
    actions = [action for action in action_str.split('\n') if action]
    for action in actions:
        try:
            action_type, params = parse_tool(action)
            matched_params = re.match(params_regex[action_type], params)
            eval_params = list(map(literal_eval, matched_params.groups()))
            trajectories.update(eval(camel_to_snake(action_type))(*eval_params, output_format=output_format))
        except Exception as e:
            logging.error(f"Failed to execute action: {action}\n{e}")
    return trajectories
    
def accommodation_search(city, filters, output_format):

    if output_format == 'planning':
        if len(filters) == 0:
            function_name = "Accommodations in {city}".format(city=city)
        else:
            function_name = tool_descriptions['accommodation'].format(city=city, filters=', '.join(filters))
        result = accommodation.search(city, filters)
        
    else:
        function_name = tool_templates['accommodation'].format(city=city, filters=json.dumps(filters))
        result = accommodation.view(city, filters) if output_format =='clarification' else None
        
    return {function_name: result}

def restaurant_search(city, preferences, output_format):
    
    if output_format == 'planning':
        if len(preferences) == 0:
            function_name = "List all restaurants in {city}".format(city=city)
        else:
            function_name = tool_descriptions['restaurant'].format(city=city, preferences=', '.join(preferences))
        result = restaurant.search(city, preferences)
    
    else: 
        function_name = tool_templates['restaurant'].format(city=city, preferences=json.dumps(preferences))
        result = restaurant.view(city, preferences) if output_format == 'clarification' else None
        
    return {function_name: result}

def attraction_search(city, output_format):
    
    if output_format == 'planning':
        function_name = tool_descriptions['attraction'].format(city=city)
        result = attraction.search(city)
    else:
        function_name = tool_templates['attraction'].format(city=city)
        result = attraction.view(city) if output_format == 'clarification' else None
        
    return {function_name: result}


def distance_matrix(origin, destination, mode, output_format):
    
    if output_format == 'planning':
        function_name = tool_descriptions['distance_matrix'].format(mode=mode, origin=origin, destination=destination)
        result = google_distance_matrix.search(origin, destination, mode)
    else:
        function_name = tool_templates['distance_matrix'].format(origin=origin, destination=destination, mode=mode)
        result = google_distance_matrix.view(origin, destination, mode) if output_format == 'clarification' else None
        
    return {function_name: result}

def flight_search(origin, destination, date, output_format):
    if output_format == 'planning':
        function_name = tool_descriptions['flight'].format(origin=origin, destination=destination, date=date)
        result = flight.search(origin, destination, date)
        
    else:
        function_name = tool_templates['flight'].format(origin=origin, destination=destination, date=date)
        result = flight.view(origin, destination, date) if output_format == 'clarification' else None
        
    return {function_name: result}

def budget_estimator(org, dest, dates, people_number, output_format):
    
    if output_format == 'planning':
        function_name = tool_descriptions['budget'].format(people_number=people_number, org=org, dest=', '.join(dest), dates=', '.join(dates))
        result = budget.search(org, dest, dates, people_number)
        
    else:
        function_name = tool_templates['budget'].format(org=org, dest=json.dumps(dest), dates=json.dumps(dates), people_number=people_number)
        result = budget.view(org, dest, dates, people_number) if output_format == 'clarification' else None
        
    return {function_name: result}

def check_valid_trajectories(function_names: List[str], transportation: str, output_format: str):
    
    function_name_set = set()
    if output_format == 'planning':
        required_function_keywords = set(["accommodation", "restaurant", "attraction", "taxi", "self-driving", "flight", "budget"])
        if transportation == 'no flight':
            required_function_keywords.remove('flight')
        elif transportation == 'no self-driving':
            required_function_keywords.remove('self-driving')
        
        for function in function_names:
            for keyword in required_function_keywords:
                if keyword in function.lower():
                    function_name_set.add(keyword)
                    break
            else:
                logging.error("Invalid tool call: {}".format(function))
                return False
            
        if function_name_set != required_function_keywords:
            logging.error("Missing tool calls: {}".format(required_function_keywords - function_name_set))
            return False
        
    else:
        for function in function_names:
            function_name, params = parse_tool(function)
            if function_name is None or params is None:
                logging.error("Invalid tool call: {}".format(function))
                return False
            function_name_set.add(function_name)
            
        if output_format == 'brute_force':
            required_function_names = set([function.split('(')[0] for function in tool_templates.values()])
            if function_name_set != required_function_names:
                logging.error("Missing tool calls: {}".format(required_function_names - function_name_set))
                return False
        
    return True
        
        
    
def generate_trajectories(data: Dict, output_format: str):
    
    assert output_format in ['planning', 'clarification', 'brute_force', 'finetuning'], "Invalid output format"
    
    org = data['org']
    cities = extract_cities(data['dest'])
    days = data['days']
    visiting_days = extract_days(data['dest']) + [days]
    people_number = data['people_number']
    housing = data['housing']
    cuisine = data['cuisine']
    transportation = data['transportation']
    
    if len(data['departure_date']) != 0 and days != -1 and len(cities) != 0:
        dates = [datetime.datetime.strptime(data['departure_date'], "%Y-%m-%d") + datetime.timedelta(days=day-1) for day in visiting_days]
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        
    elif output_format == 'brute_force':
        if len(data['departure_date']) == 0:
            dates = ["na"] * max(len(visiting_days), 2)
        elif len(cities) == 0 and days == -1:
            dates = [data['departure_date'], "na"]
        elif len(cities) == 0:
            return_date = datetime.datetime.strptime(data['departure_date'], "%Y-%m-%d") + datetime.timedelta(days=days-1)
            dates = [data['departure_date'], return_date.strftime("%Y-%m-%d")]
        elif days == -1:
            dates = [datetime.datetime.strptime(data['departure_date'], "%Y-%m-%d") + datetime.timedelta(days=day-1) for day in visiting_days[:-1]]
            dates = [date.strftime("%Y-%m-%d") for date in dates] + ["na"]
        else:
            raise ValueError("In the brute force mode, the situation where the departure date is not provided and the number of days is not provided should be handled earlier.")
    else:
        dates = []
           
    if output_format == 'brute_force':
        cities = ["na"] if len(cities) == 0 else cities
        people_number = 0 if people_number == -1 else people_number
        org = "na" if len(org) == 0 else org
    
    trajectories = {}
    for city in cities:
        trajectories.update(accommodation_search(city, housing, output_format))
        trajectories.update(restaurant_search(city, cuisine, output_format))
        trajectories.update(attraction_search(city, output_format))

    destinations = cities.copy()
    cities = [org] + cities + [org]
    if len(org) > 0:
        for i in range(len(cities) - 1):
            if len(transportation) == 0 or output_format == "brute_force":
                trajectories.update(distance_matrix(cities[i], cities[i + 1], 'taxi', output_format))
                trajectories.update(distance_matrix(cities[i], cities[i + 1], 'self-driving', output_format))
                if len(dates) > 0:
                    trajectories.update(flight_search(cities[i], cities[i + 1], dates[i], output_format))
                
            elif transportation == 'no flight':
                trajectories.update(distance_matrix(cities[i], cities[i + 1], 'taxi', output_format))
                trajectories.update(distance_matrix(cities[i], cities[i + 1], 'self-driving', output_format))
                
            elif transportation == 'no self-driving':
                trajectories.update(distance_matrix(cities[i], cities[i + 1], 'taxi', output_format))
                if len(dates) > 0:
                    trajectories.update(flight_search(cities[i], cities[i + 1], dates[i], output_format))
                
            else:
                raise ValueError("Invalid transportation mode")
    
    if len(org) != 0 and len(destinations) != 0 and len(dates) != 0 and people_number != -1:
        trajectories.update(budget_estimator(org, destinations, dates, people_number, output_format))

    assert check_valid_trajectories(list(trajectories.keys()), transportation, output_format)
    return list(trajectories.keys()) if output_format == 'brute_force' or output_format == 'finetuning' else trajectories