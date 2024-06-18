import pandas as pd
from pandas import DataFrame
import os, sys
from typing import Tuple
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database/restaurants/clean_restaurant_2022.csv"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import validate_city_format, CuisineError, to_string

class Restaurants:
    def __init__(self, path=data_path):
        self.path = path
        self.cuisine_types = ["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]
        self.data = pd.read_csv(self.path).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
        print("Restaurants loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == city]
        # results = results[results["date"] == date]
        # if price_order == "asc":
        #     results = results.sort_values(by=["Average Cost"], ascending=True)
        # elif price_order == "desc": 
        #     results = results.sort_values(by=["Average Cost"], ascending=False)

        # if rating_order == "asc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=True)
        # elif rating_order == "desc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=False)
        if len(results) == 0:
            return "There is no restaurant in this city."
        return results
    
    def query(self, city: str, cuisines: list) -> Tuple:
        if validate_city_format([city]):
            results = self.data[self.data["City"] == city]
            message = ""
            if len(cuisines) > 0:
                for cuisine in cuisines:
                    if cuisine not in self.cuisine_types:
                        raise CuisineError(cuisine)
                    if results[results["Cuisines"].str.contains(cuisine, case=False, na=False)].empty:
                        message += f"\nThere is no {cuisine} restaurant in this city."
            return results, message
            
    def search(self, city: str, cuisines: list) -> str:
        results, message = self.query(city, cuisines)
        if len(results) == 0:
            return "There is no restaurant in this city."
        if len(message) > 0:
            return to_string(results) + message
        else:
            return to_string(results)
            
    def view(self, city: str, cuisines: list) -> str:
        results, message = self.query(city, cuisines)
        if len(results) == 0:
            return "There is no restaurant in this city."
        if len(message) > 0:
            return to_string(results, verbose=False) + message
        else:
            return to_string(results, verbose=False)