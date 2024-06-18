import pandas as pd
from pandas import DataFrame
import re
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import validate_city_format, to_string

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database/attractions/attractions.csv"))

class Attractions:
    def __init__(self, path=data_path):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results
    
    def query(self, city: str) -> str:
        if validate_city_format([city]):
            return self.run(city)
        
    def search(self, city: str) -> str:
        return to_string(self.query(city))
    
    def view(self, city: str) -> str:
        return to_string(self.query(city), verbose=False)
    