import datetime, math, os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import validate_date_format, validate_city_format, to_string, PeopleNumberError

def separate_numbers(number):
    # add , to the number per 3 digits using loop
    number = str(number)
    if len(number) <= 3:
        return number
    result = ''
    for i in range(len(number)):
        if i % 3 == 0 and i != 0:
            result += ','
        result += number[-i-1]
    return result[::-1]

def validate_people_number(people_number):
    if isinstance(people_number, int) == False or people_number <= 0:
        return PeopleNumberError(people_number)
    return True

def calc_taxi_cost(taxi_data, people_number):
    return taxi_data * math.ceil(people_number * 1.0 / 4)


def calc_self_driving_cost(self_driving_data, people_number):
    return self_driving_data * math.ceil(people_number * 1.0 / 5)


def calc_flight_cost(flight_data, people_number):
    return min(flight_data["Price"].tolist()) * people_number


def check_flight_result(flight_data):
    if type(flight_data) == str and "There is no flight" in flight_data:
        return False
    return True


def calc_transportation(flight_data, taxi_data, self_driving_data, people_number):
    total_cost = 0
    self_driving_fee, no_self_driving_fee = 0, 0
    self_driving_status, no_self_driving_status = True, True
    for flight_info, taxi_info, self_driving_info in zip(
        flight_data, taxi_data, self_driving_data
    ):
        flight_result = check_flight_result(flight_info)

        if flight_result == False and taxi_info == None and self_driving_info == None:
            return -1
        if flight_result == False and taxi_info == None:
            no_self_driving_status = False
        else:
            minimal_cost = 0x7FFFFFFF
            if flight_result:
                minimal_cost = min(
                    minimal_cost, calc_flight_cost(flight_info, people_number)
                )
            if taxi_info != None:
                minimal_cost = min(
                    minimal_cost, calc_taxi_cost(taxi_info, people_number)
                )
            assert (
                minimal_cost != 0x7FFFFFFF
            ), "No valid information for transportation."
            no_self_driving_fee += minimal_cost

        if self_driving_info != None:
            self_driving_fee += calc_self_driving_cost(self_driving_info, people_number)
        else:
            self_driving_status = False

    if self_driving_status or no_self_driving_status:
        if self_driving_status and no_self_driving_status:
            total_cost = min(self_driving_fee, no_self_driving_fee)
        elif self_driving_status:
            total_cost = self_driving_fee
        elif no_self_driving_status:
            total_cost = no_self_driving_fee

        return total_cost
    else:
        return -1


def calc_hotels(hotel_data, durations, people_number):
    total_cost = 0
    for hotel_info, duration in zip(hotel_data, durations):
        if hotel_info.empty:
            return -1
        min_cost = (
            hotel_info["price"]
            * hotel_info.apply(
                lambda row: math.ceil(people_number * 1.0 / row["maximum occupancy"]),
                axis=1,
            )
        ).min()
        if pd.isna(min_cost):
            return -1
        total_cost += min_cost * duration.days

    return total_cost


class Budget:
    def __init__(self, hotel_api, flight_api, distance_matrix_api):
        self.hotel = hotel_api
        self.flight = flight_api
        self.distance_matrix = distance_matrix_api
        print("BudgetEstimator loaded.")
        
    def run(self, org: str, dest: list, dates: list, people_number: int):
        """
        Run the budget estimator tool.
        """

        all_hotel_data = []
        all_flight_data = []
        all_taxi_data = []
        all_self_driving_data = []

        durations = []
        for idx in range(1, len(dates)):
            
            durations.append(dates[idx] - dates[idx - 1])

        # print(org, city_list, dates, durations)
        assert len(durations) == len(
            dest
        ), "The number of destinations should be one less than the number of dates."

        last_city = org
        for idx, (current_city, duration) in enumerate(zip(dest, durations)):

            # Fetch data for the current city
            current_hotel_data = self.hotel.run_for_evaluation(current_city, duration.days)
            current_flight_data = self.flight.run(
                last_city, current_city, dates[idx].strftime("%Y-%m-%d")
            )
            current_taxi_data = self.distance_matrix.run_for_evaluation(
                last_city, current_city, "taxi"
            )["cost"]
            current_self_driving_data = self.distance_matrix.run_for_evaluation(
                last_city, current_city, "driving"
            )["cost"]

            last_city = current_city
            # Append the dataframes to the lists
            all_hotel_data.append(current_hotel_data)
            all_flight_data.append(current_flight_data)
            all_taxi_data.append(current_taxi_data)
            all_self_driving_data.append(current_self_driving_data)

        return_flight_data = self.flight.run(last_city, org, dates[-1].strftime("%Y-%m-%d"))
        return_taxi_data = self.distance_matrix.run_for_evaluation(last_city, org, "taxi")[
            "cost"
        ]
        return_self_driving_data = self.distance_matrix.run_for_evaluation(
            last_city, org, "driving"
        )["cost"]
        all_flight_data.append(return_flight_data)
        all_taxi_data.append(return_taxi_data)
        all_self_driving_data.append(return_self_driving_data)

        transportation_budget = calc_transportation(
            all_flight_data, all_taxi_data, all_self_driving_data, people_number
        )
        hotel_budget = calc_hotels(all_hotel_data, durations, people_number)
        total_budget = math.floor(transportation_budget + hotel_budget)

        if hotel_budget <= 0:
            return "No valid hotel information is available to estimate the minimum budget."
        elif transportation_budget <= 0:
            return "No valid transportation information is available to estimate the minimum budget."
        else:
            return "$" + separate_numbers(total_budget)
    
    def search(self, org, dest, dates, people_number):
        if validate_city_format([org] + dest) and validate_date_format(dates) and validate_people_number(people_number):
            for idx, date in enumerate(dates):
                dates[idx] = datetime.datetime.strptime(date, "%Y-%m-%d")
            return to_string(self.run(org, dest, dates, people_number))

    def view(self, org, dest, dates, people_number):
        return self.search(org, dest, dates, people_number)