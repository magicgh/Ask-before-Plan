tool_description = """1. AccommodationSearch(city, filters)
- Description: Discover accommodations in your desired city with specific filters.
- Parameters: 
  - city (str, required): The name of the city where you're seeking accommodation.
  - filters (list[str], required): A list of filters to refine your search. Choices include "shared room", "not shared room", "private room", "entire room", "parties", "smoking", "children under 10", "pets", "visitors". If the user does not specify any accommodation filters, assign an empty list "[]" to the parameter.

2. RestaurantSearch(city, cuisines)
- Description: List all restaurants in your chosen city, regardless of cuisine type, and check if any of the cuisines you specify are unavailable there.
- Parameters:
  - city (str, required): The name of the city where you're seeking restaurants.
  - cuisines (list[str], required): A list of desired cuisines to check for availability. Available options include "Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", and "French". If the user does not specify any cuisines, assign an empty list "[]" to the parameter.

3. AttractionSearch(city)
- Description: Find attractions in a city of your choice.
- Parameters: 
  - city (str, required): The name of the city where you're seeking attractions.

4. DistanceMatrix(origin, destination, mode)
- Description: Estimate the distance, time, and cost between two cities.
- Parameters:
  - origin (str, required): The departure city of your journey.
  - destination (str, required): The destination city of your journey.
  - mode (str, required): The method of transportation. Choices include "self-driving" and "taxi".

5. FlightSearch(origin, destination, date):
- Description: A flight information retrieval tool.
- Parameters:
  - origin (str, required): The city you'll be flying out from.
  - destination (str, required): The city you aim to reach.
  - date (str, required): The date of your travel in "YYYY-MM-DD" format.

6. BudgetEstimator(origin, destination, dates, people_number)
- Description: Calculate the minimal estimated budget for the trip. Use this tool to verify if the budget provided by the user is sufficient.
- Parameters:
  - origin (str, required): The departure city of your trip.
  - destination (list[str], required): A list of cities you plan to visit during your trip.
  - dates (list[str], required): A list of dates corresponding to the departure from the origin and each of the destinations. The first date is the departure from the origin, and each subsequent date corresponds to the departure from the respective city in the destination list. The last date in this list is the departure from the final destination back to the origin city. All dates should be formatted as "YYYY-MM-DD".
  - people_number (int, required): The number of people on the trip."""

tool_description_with_example = ["""1. AccommodationSearch(city, filters)
- Description: Discover accommodations in your desired city with specific filters.
- Parameters: 
  - city (str, required): The name of the city where you're seeking accommodation.
  - filters (list[str], required): A list of filters to refine your search. Choices include "shared room", "not shared room", "private room", "entire room", "parties", "smoking", "children under 10", "pets", "visitors". If the user does not specify any accommodation filters, assign an empty list "[]" to the parameter.
- Example: AccommodationSearch("Berlin", ["private room", "parties"]) would return private rooms in Berlin that allow parties.""",

"""2. RestaurantSearch(city, cuisines)
- Description: List all restaurants in your chosen city, regardless of cuisine type, and check if any of the cuisines you specify are unavailable there.
- Parameters:
  - city (str, required): The name of the city where you're seeking restaurants.
  - cuisines (list[str], required): A list of desired cuisines to check for availability. Available options include "Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", and "French". If the user does not specify any cuisines, assign an empty list "[]" to the parameter.
- Example: RestaurantSearch("Dublin", ["Chinese", "Italian", "French"]) returns all restaurants in Dublin and informs you if any of the Chinese, Italian, or French cuisines are unavailable.""",

"""3. AttractionSearch(city)
- Description: Find attractions in a city of your choice.
- Parameters: 
  - city (str, required): The name of the city where you're seeking attractions.
- Example: AttractionSearch("London") would return attractions in London.""",

"""4. DistanceMatrix(origin, destination, mode)
- Description: Estimate the distance, time, and cost between two cities.
- Parameters:
  - origin (str, required): The departure city of your journey.
  - destination (str, required): The destination city of your journey.
  - mode (str, required): The method of transportation. Choices include "self-driving" and "taxi".
- Example: DistanceMatrix("Paris", "Lyon", "self-driving") would provide driving distance, time, and cost between Paris and Lyon.""",

"""5. FlightSearch(origin, destination, date):
- Description: A flight information retrieval tool.
- Parameters:
  - origin (str, required): The city you'll be flying out from.
  - destination (str, required): The city you aim to reach.
  - date (str, required): The date of your travel in "YYYY-MM-DD" format.
- Example: FlightSearch("New York", "London", "2022-10-01") would fetch flights from New York to London on October 1, 2022.""",

"""6. BudgetEstimator(origin, destination, dates, people_number)
- Description: Calculate the minimal estimated budget for the trip. Use this tool to verify if the budget provided by the user is sufficient.
- Parameters:
  - origin (str, required): The departure city of your trip.
  - destination (list[str], required): A list of cities you plan to visit during your trip.
  - dates (list[str], required): A list of dates corresponding to the departure from the origin and each of the destinations. The first date is the departure from the origin, and each subsequent date corresponds to the departure from the respective city in the destination list. The last date in this list is the departure from the final destination back to the origin city. All dates should be formatted as "YYYY-MM-DD".
  - people_number (int, required): The number of people on the trip.
- Example: BudgetEstimator("London", ["Paris", "Rome", "Madrid"], ["2022-09-01", "2022-09-05", "2022-09-10", "2022-09-15"], 2) would return the minimal estimated budget for a trip from London to Paris, from Paris to Rome, from Rome to Madrid, and from Madrid back to London on September 1, 5, 10, and 15, 2022, respectively, for two people."""]

finish_description_with_example = """7. Finish()
- Description: Use this function to indicate the task's completion once all the necessary information has been collected.
- Example: Call Finish() after gathering all necessary information related to accommodations, dining, attractions, transportation, and budget."""
