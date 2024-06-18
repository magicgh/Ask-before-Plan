import sys, json, os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))

from common.data import load_data, extract_cities, extract_days, expand_task, load_results, write_results, precheck_path
from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
import math
from tqdm import tqdm
import re
import random
import argparse
import datetime
from filelock import FileLock

flight = Flights()
accommodations = Accommodations()
restaurants = Restaurants()
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions()


def load_line_json_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            unit = json.loads(line)
            data.append(unit)
    return data


def extract_before_parenthesis(s):
    match = re.search(r"^(.*?)\([^)]*\)", s)
    return match.group(1) if match else s


def get_transportation(org, dest, date):
    transportation_price_info = {"Flight": 1e9, "Self-driving": 1e9, "Taxi": 1e9}
    # get the flight information
    flight_info = flight.run(org, dest, date)
    if type(flight_info) != str and len(flight_info) > 0:
        flight_cost = int(
            flight_info.sort_values(by=["Price"], ascending=True).iloc[0]["Price"]
        )
        transportation_price_info["Flight"] = flight_cost
    # get the self-driving information
    self_driving_info = googleDistanceMatrix.run_for_evaluation(
        org, dest, mode="driving"
    )
    if self_driving_info["cost"] != None:
        transportation_price_info["Self-driving"] = self_driving_info[
            "cost"
        ] * math.ceil(1.0 / 5)
    # get the taxi information
    taxi_info = googleDistanceMatrix.run_for_evaluation(org, dest, mode="taxi")
    if taxi_info["cost"] != None:
        transportation_price_info["Taxi"] = taxi_info["cost"] * math.ceil(1.0 / 4)
    sorted_dict = dict(
        sorted(transportation_price_info.items(), key=lambda item: item[1])
    )
    transportation = list(sorted_dict.keys())[0]
    if transportation_price_info[transportation] == 1e9:
        return False, None
    if transportation == "Flight":
        transportation = f"Flight Number: {flight_info.sort_values(by=['Price'],ascending=True).iloc[0]['Flight Number']}"
    return True, transportation


def get_meal(city):
    restaurant = restaurants.run(city)
    if type(restaurant) == str:
        return False, None
    restaurant = restaurant.sort_values(by=["Average Cost"], ascending=True)

    for idx in range(len(restaurant)):
        # if f"{restaurant.iloc[idx]['Name']}, {city}" not in restaurant_data_list:
        return True, f"{restaurant.iloc[idx]['Name']}, {city}"
    return False, None


def get_attraction(city):
    attraction = attractions.run(city)
    if type(attraction) == str:
        return False, None
    idx = random.choice([i for i in range(len(attraction))])
    return True, f"{attraction.iloc[idx]['Name']}, {city}"
    # return False, None


def get_accommodation(city):
    accommodation = accommodations.run(city)

    if type(accommodation) == str:
        return False, None
    accommodation = accommodation.sort_values(by=["price"], ascending=True)
    if len(accommodation) == 0:
        return False, None

    return True, f"{accommodation.iloc[0]['NAME']}, {city}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    args = parser.parse_args()
    output_path = precheck_path(os.path.join(args.output_dir, args.data_split))

    cleaned_data = load_data(split=args.data_split, start_idx=args.start_idx, end_idx=args.end_idx)

    for data_idx, sample in enumerate(tqdm(cleaned_data)):
            
        _, query = list(expand_task(sample))[-1]

        plan_list = [{"finished": [False, set()]}]
        restaurant_list = []
        attraction_list = []
        finished = False

        city_list = [query["org"]] + extract_cities(query["dest"]) + [query["org"]]
        day_list = extract_days(query["dest"]) + [query["days"]]

        dates = [
            datetime.datetime.strptime(query["departure_date"], "%Y-%m-%d")
            + datetime.timedelta(days=day - 1)
            for day in day_list
        ]
        dates = [date.strftime("%Y-%m-%d") for date in dates]

        for current_day in range(1, query["days"] + 1):
            plan = {
                key: "-"
                for key in [
                    "day",
                    "current_city",
                    "transportation",
                    "breakfast",
                    "lunch",
                    "dinner",
                    "attraction",
                    "accommodation",
                ]
            }
            plan["day"] = current_day
            current_city = None

            if current_day in day_list:
                idx = day_list.index(current_day)
                plan["current_city"] = f"from {city_list[idx]} to {city_list[idx+1]}"
                # get the transportation information
                flag, transportation = get_transportation(
                    city_list[idx], city_list[idx + 1], dates[idx]
                )
                if flag:
                    plan["transportation"] = (
                        f"{transportation}, from {city_list[idx]} to {city_list[idx+1]}"
                    )
                else:
                    plan_list[0]["finished"][0] = False
                    plan_list[0]["finished"][1].add(
                        "No valid transportation information."
                    )

            if plan["current_city"] == "-":
                if " to " not in plan_list[-1]["current_city"]:
                    plan["current_city"] = plan_list[-1]["current_city"]
                else:
                    plan["current_city"] = plan_list[-1]["current_city"].split(" to ")[1]
                current_city = plan["current_city"]
            else:
                current_city = plan["current_city"].split(" to ")[1]

            # print(current_city)
            for key in ["breakfast", "lunch", "dinner"]:
                flag, meal = get_meal(current_city)
                if flag:
                    plan[key] = f"{meal}"
                    restaurant_list.append(meal)
                else:
                    plan_list[0]["finished"][0] = False
                    plan_list[0]["finished"][1].add("No valid meal information.")

            flag, attraction = get_attraction(current_city)
            if flag:
                plan["attraction"] = f"{attraction}"
            else:
                plan_list[0]["finished"][0] = False
                plan_list[0]["finished"][1].add("No valid attraction information.")

            if current_day != query["days"]:
                flag, accommodation = get_accommodation(current_city)
                if flag:
                    plan["accommodation"] = f"{accommodation}"
                else:
                    plan_list[0]["finished"][0] = False
                    plan_list[0]["finished"][1].add(
                        "No valid accommodation information."
                    )

            plan_list.append(plan)

        if plan_list[0]["finished"][1] == set():
            plan_list[0]["finished"] = (True, [])
        
        save_path = os.path.join(output_path, f"task_{args.start_idx + data_idx}.json")
        with FileLock(save_path + '.lock'):
            generated_result = load_results(save_path, "plan")
                    
            generated_result["plan"]["greedy_search_log"] = [
                plan_list[0]["finished"][0],
                list(plan_list[0]["finished"][1]),
            ]
            generated_result["plan"]["greedy_search_results"] = plan_list[1:]
            # print(generated_result[-1])
            write_results(save_path, generated_result)
