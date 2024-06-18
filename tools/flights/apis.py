import pandas as pd
from pandas import DataFrame
from typing import Optional
import re, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import validate_city_format, validate_date_format, to_string

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database/flights/clean_Flights_2022.csv"))

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

class Flights:

    def __init__(self, path=data_path):
        self.path = path
        self.data = None

        self.data = pd.read_csv(self.path).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime','FlightDate','OriginCityName','DestCityName','Distance']]
        print("Flights API loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna().rename(columns={'Unnamed: 0': 'Flight Number'})

    def run(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        results = results[results["FlightDate"] == departure_date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        if len(results) == 0:
            return "There is no flight from {} to {} on {}.".format(origin, destination, departure_date)
        return results
    
    def query(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> str:
        if validate_city_format([origin, destination]) and validate_date_format([departure_date]):
            return self.run(origin, destination, departure_date)
        
    def search(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> str:
            return to_string(self.query(origin, destination, departure_date))
    
    def view(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> str:
            return to_string(self.query(origin, destination, departure_date), verbose=False)
    
    def run_evaluate(self,
            origin: str,
            destination: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        if len(results) == 0:
            return "There is no flight from {} to {}.".format(origin, destination)
        return results
    
    def run_for_annotation(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == extract_before_parenthesis(origin)]
        results = results[results["DestCityName"] == extract_before_parenthesis(destination)]
        results = results[results["FlightDate"] == departure_date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        return results.to_string(index=False)

    def get_city_set(self):
        city_set = set()
        for unit in self.data['data']:
            city_set.add(unit[5])
            city_set.add(unit[6])