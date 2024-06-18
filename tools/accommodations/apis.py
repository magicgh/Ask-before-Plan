import pandas as pd
from pandas import DataFrame
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import validate_city_format, FilterError, to_string

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database/accommodations/clean_accommodations_2022.csv"))

class Accommodations:
    def __init__(self, path=data_path):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        self.room_types = {
            "private room": ["Private room"],
            "entire room": ["Entire home/apt"],
            "not shared room": ["Private room", "Entire home/apt"],
            "shared room": ["Shared room"],
        }
        self.house_rules = ["smoking", "parties", "children under 10", "visitors", "pets"]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run_for_evaluation(self, city: str, duration: int) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        results = results[results["minimum nights"] <= duration]
        return results
    
    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        if len(results) == 0:
            return "There is no accommodation in this city."
        
        return results
    
    def query(self, city: str, keywords: list) -> str:
        
        if validate_city_format([city]):
            results = self.data[self.data["city"] == city]
            
            for keyword in keywords:
                if keyword in self.room_types:
                    results = results[results["room type"].isin(self.room_types[keyword])]
                elif keyword in self.house_rules:
                    results = results[~results["house_rules"].str.contains(f"No {keyword}", case=False, na=False)]
                else:
                    raise FilterError(keyword)
                
            if len(results) == 0:
                return "There is no accommodation in this city."
            else:
                return results
    
    def search(self, city: str, keywords: list) -> str:
        
        return to_string(self.query(city, keywords))
    
    def view(self, city: str, keywords: list) -> str:
        
        return to_string(self.query(city, keywords), verbose=False)