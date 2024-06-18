import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from collections import defaultdict
from template import tool_description_with_example, finish_description_with_example
tool_prompts = defaultdict(dict)

expanded_tool_description_with_example = "\n\n".join(tool_description_with_example)

tool_prompts["direct"]["system"] = f"""Based on the provided conversation between the user and the assistant, generate function calls to collect valid information related to accommodations, dining, attractions, transportation, and budget. The available functions are detailed below:

{expanded_tool_description_with_example}

Please ensure that nested function use is avoided, escape symbols are not included in the string, and functions are only called when all required parameters are available. Your response should include all available function calls, specifying both the function name and its parameters, with each function on a separate line."""

tool_prompts["direct"]["user"] = "Conversation: {conversations}\nResponse:"

tool_prompts["direct"]["example_user"] = "Conversation: [{'user': 'Could you create a 3-day travel plan for 7 people from Ithaca to Portland on day 1, from March 8th, 2022?'}, {'assistant': 'Sorry, I couldn\'t find a way to arrive in Portland. Could you provide an alternative destination?'}, {'user': 'Charlotte.'}]\nResponse:"

tool_prompts["direct"]["example_agent"] = 'AccommodationSearch("Charlotte", [])\nRestaurantSearch("Charlotte", [])\nAttractionSearch("Charlotte")\nDistanceMatrix("Ithaca", "Charlotte", "taxi")\nDistanceMatrix("Ithaca", "Charlotte", "self-driving")\nFlightSearch("Ithaca", "Charlotte", "2022-03-08")\nDistanceMatrix("Charlotte", "Ithaca", "taxi")\nDistanceMatrix("Charlotte", "Ithaca", "self-driving")\nFlightSearch("Charlotte", "Ithaca", "2022-03-10")\nBudgetEstimator("Ithaca", ["Charlotte"], ["2022-03-08", "2022-03-10"], 7)'

tool_prompts["react"]["system"] = f"""Based on the provided conversation between the user and the assistant, collect valid information related to accommodations, dining, attractions, transportation, and budget. Solve this task by alternating between "Thought", "Action", and "Observation" steps. "Thought" can reason about the current situation, and "Action" can have 7 different types:

{expanded_tool_description_with_example}

{finish_description_with_example}

Please ensure that nested function use is avoided, escape symbols are not included in the string, and functions are only called when all required parameters are available. Each action should call a single function once, using the valid function name and all required parameters. You should take as many steps as possible until you have gathered the necessary information to complete the task using Finish(). If the user's request is vague or infeasible, avoid making assumptions and strictly use the provided information. Do not add any description or comment to the action. Additionally, do not include line breaks in your response."""

tool_prompts["react"]["user"] = "Conversation: {conversations}\n{scratchpad}"

tool_categories = [
    "accommodations", "restaurants", "attractions", 
    "googleDistanceMatrix", "flights", "budget"
]

# Constructing the API documentation dictionary
tool_description_set = {category: tool_description_with_example[i] for i, category in enumerate(tool_categories)}

# Adding the finish description with example
tool_description_set["finish"] = finish_description_with_example

tool_prompts["reflect"]["system"] = tool_prompts["react"]["system"]
tool_prompts["reflect"]["user"] = "{reflections}\n" + tool_prompts["react"]["user"]
tool_prompts["reflect"]["header"] = "In previous attempts, you tried to use tools to interact with the environment to gather valid information on accommodations, dining, attractions, transportation, and budget given the user-assistant conversation but were unsuccessful. The reflections below offer suggestions to help you avoid past mistakes. Use these insights to refine your strategy for effectively and efficiently utilizing tools to collect the necessary information."
tool_prompts["reflect"]["reflection"] = """You are an advanced reasoning agent capable of self-improvement through reflection. You will review a previous attempt where you failed to effectively utilize a tool to gather valid information about accommodations, dining, attractions, transportation, and budget given the user-assistant conversation. Analyze the reasons for the mistake, referencing the tool documentation, the observation, and the action you have taken. Then, formulate a concise, high-level explanation to address and prevent similar errors in the future. Keep your reflections in complete sentences without any line breaks.

Tool documentation:
{tool_docs}
Ensure that each action uses only one non-nested tool and contains no comments or descriptions.

Observation: {observation}
Action: {action}

Reflection:"""

error_prompts = {}
error_prompts["invalid_action"] = 'Invalid action: "{action}". Valid actions include AccommodationSearch(city, filters), RestaurantSearch(city, cuisines), AttractionSearch(city), DistanceMatrix(origin, destination, mode), FlightSearch(origin, destination, date), BudgetEstimator(origin, destination, dates, people_number), and Finish(). Do not include any comment or description in the action.'

error_prompts["invalid_params"] = "Invalid parameters for {action}. Please ensure that all parameters are provided in the correct format."

error_prompts["null_action"] = "Your action has been filtered due to content restrictions. Please ensure your action does not begin with ['\\n', 'Thought', 'Action', 'Observation']. Ensure that the action is permitted in this environment, and try again."

error_prompts["no_feedback"] = "No feedback from the environment due to the null action. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation']."