from collections import defaultdict
planner_prompts = defaultdict(dict)

base_system_message = 'You are a proficient planner tasked with generating a detailed travel plan in JSON format, which is an array of objects, based on the interaction trajectory and the user-assistant conversation. Your plan must strictly adhere to the format provided in the example, incorporating specific details such as flight numbers (e.g., "F0123456"), restaurant names, and accommodation names. Ensure all information in your plan is derived solely from the provided data and aligns with common sense. Attraction visits and meals are expected to be diverse. Use the symbol "-" for unnecessary details, such as "eat at home" or "on the road". For instance, you do not need to plan after returning to the departure city. When traveling to two cities in one day, ensure that "current_city" aligns with the format "from A to B" in the example. If transportation details indicate a journey from one city to another (e.g., from A to B), update the "current_city" to the destination city (in this case, B) the following day. Use ";" to separate different attractions, formatting each as "Name, City". Make sure all flight numbers and costs are appended with a colon (e.g., "Flight Number:" and "Cost:"), consistent with the example. Your JSON plan should include the following fields: ["day", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]. Escape symbols should be used in the string when necessary. Additionally, remove any "$" symbols and comments from the plan.'
planner_prompts["direct"]["system"] = (
    base_system_message
    + "\n\n"
    + """***** Example *****
Conversation: [{'user': 'Could you create a 3-day travel plan for 7 people from Ithaca to Portland on day 1, from March 8th, 2022?'}, {'assistant': 'Sorry, I couldn\'t find a way to arrive in Portland. Could you provide an alternative destination?'}, {'user': 'Charlotte.'}, {'assistant': 'It seems you haven\'t mentioned the expected budget for this trip. Could you provide that information?'}, {'user': 'Yes, my expected budget is $30,200.'}]
Travel Plan: [{"day": 1, "current_city": "from Ithaca to Charlotte", "transportation": "Flight Number: F3633405, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 08:10", "breakfast": "Nagaland\'s Kitchen, Charlotte", "attraction": "The Charlotte Museum of History, Charlotte;", "lunch": "Cafe Maple Street, Charlotte", "dinner": "Bombay Vada Pav, Charlotte", "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}, {"day": 2, "current_city": "Charlotte", "transportation": "-", "breakfast": "Olive Tree Cafe, Charlotte", "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte;", "lunch": "Birbal Ji Dhaba, Charlotte", "dinner": "Pind Balluchi, Charlotte", "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}, {"day": 3, "current_city": "from Charlotte to Ithaca", "transportation": "Flight Number: F3786160, from Charlotte to Ithaca, Departure Time: 20:48, Arrival Time: 22:34", "breakfast": "Subway, Charlotte", "attraction": "Books Monument, Charlotte;", "lunch": "Olive Tree Cafe, Charlotte", "dinner": "Kylin Skybar, Charlotte", "accommodation": "-"}]
***** Example Ends *****"""
)

planner_prompts["direct"]["user"] = "Interaction trajectory: {trajectories}\nConversation: {conversations}\nTravel Plan:"

planner_prompts["cot"]["system"] = planner_prompts["direct"]["system"]

planner_prompts["cot"]["user"] = planner_prompts["direct"]["user"] + " Let's think step by step. First,"

planner_prompts["react"]["system"] = (
    base_system_message
    + """ Solve this task by alternating between "Thought", "Action", and "Observation" steps. The "Thought" phase involves reasoning about the current situation. The "Action" phase can be of two types:
1. CostEnquiry(subplan): This function calculates the cost of a detailed subplan, for which you need to input the number of people and plan in JSON format. The subplan should encompass a complete one-day plan and include the following fields: ["people_number", "day", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]. An example will be provided for reference.
2. Finish(final_json_plan): Use this function to indicate the completion of the task. You must submit a final, complete plan in JSON as the argument.

***** Example *****
Conversation: [{'user': 'Could you create a 3-day travel plan for 7 people from Ithaca to Portland on day 1, from March 8th, 2022?'}, {'assistant': 'Sorry, I couldn\'t find a way to arrive in Portland. Could you provide an alternative destination?'}, {'user': 'Charlotte.'}, {'assistant': 'It seems you haven\'t mentioned the expected budget for this trip. Could you provide that information?'}, {'user': 'Yes, my expected budget is $30,200.'}]
You can call CostEnquiry like CostEnquiry({"people_number": 7, "day": 1, "current_city": "from Ithaca to Charlotte", "transportation": "Flight Number: F3633405, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 08:10", "breakfast": "Nagaland's Kitchen, Charlotte", "attraction": "The Charlotte Museum of History, Charlotte", "lunch": "Cafe Maple Street, Charlotte", "dinner": "Bombay Vada Pav, Charlotte", "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"})
You can call Finish like Finish([{"day": 1, "current_city": "from Ithaca to Charlotte", "transportation": "Flight Number: F3633405, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 08:10", "breakfast": "Nagaland\'s Kitchen, Charlotte", "attraction": "The Charlotte Museum of History, Charlotte;", "lunch": "Cafe Maple Street, Charlotte", "dinner": "Bombay Vada Pav, Charlotte", "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}, {"day": 2, "current_city": "Charlotte", "transportation": "-", "breakfast": "Olive Tree Cafe, Charlotte", "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte;", "lunch": "Birbal Ji Dhaba, Charlotte", "dinner": "Pind Balluchi, Charlotte", "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}, {"day": 3, "current_city": "from Charlotte to Ithaca", "transportation": "Flight Number: F3786160, from Charlotte to Ithaca, Departure Time: 20:48, Arrival Time: 22:34", "breakfast": "Subway, Charlotte", "attraction": "Books Monument, Charlotte;", "lunch": "Olive Tree Cafe, Charlotte", "dinner": "Kylin Skybar, Charlotte", "accommodation": "-"}])
***** Example Ends *****

You must use Finish(final_json_plan) to indicate that you have finished the task. Each action only calls one function once, without any comments or descriptions. Do not include line breaks in your response."""
)

planner_prompts["react"]["user"] = "Interaction trajectory: {trajectories}\nConversation: {conversations}\n{scratchpad}"

planner_prompts["reflect"]["system"] = planner_prompts["react"]["system"]

planner_prompts["reflect"]["user"] = "{reflections}\n" + planner_prompts["react"]["user"]

planner_prompts["reflect"]["header"] = 'You have attempted to give a subplan before and failed. The following reflection(s) give a suggestion to avoid failing to answer the query in the same way you did previously. Use them to improve your strategy for correctly planning.'

planner_prompts["reflect"]["reflection"] = """You are an advanced reasoning agent who can improve based on self-reflection. You will be given a previous reasoning trial in which you were given access to an automatic cost calculation environment, the conversation, and the previous interaction trajectory. Only the selection whose name and city match the given information will be calculated correctly. You were unsuccessful in creating a plan because you used up your set number of reasoning steps. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high-level plan that aims to mitigate the same failure. Keep your reflections in complete sentences without any line breaks.

Conversation: {conversations}
Interaction trajectory: {trajectories}
Previous trial: {scratchpad}

Reflection:"""

error_prompts = {}

error_prompts["invalid_subplan"] = "The subplan cannot be parsed into JSON format; please check. Only a one-day plan is supported."
error_prompts["error_subplan"] = "The subplan cannot be parsed into JSON format due to the syntax error; please check."
error_prompts["null_action"] = "Your action has been filtered due to content restrictions. Please ensure your action does not begin with ['\\n', 'Thought', 'Action', 'Observation']."
error_prompts["invalid_action"] = 'Invalid action. Valid actions include CostEnquiry(subplan) and Finish(final_json_plan). Please ensure that the parameter is provided in the correct format. Do not include any comments, descriptions, or line breaks in your response.'
