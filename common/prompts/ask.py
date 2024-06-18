import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from collections import defaultdict
from template import tool_description
ask_prompts = defaultdict(dict)

null_tips = " Note that if the user does not specify details regarding accommodation, cuisine, or transportation in the initial query, it indicates that the user does not have specific preferences that need clarification in the following conversation."
ask_prompts["direct"]["system"] = f"""Your current task is to determine the user's intentions and satisfy their needs based on the provided conversation between the user and the assistant, along with the interaction trajectory involving tool use between the agent and the environment. The interaction trajectory includes the following tools:

{tool_description}

If certain tools are not called in the interaction trajectory, it indicates a lack of the required parameters needed to call those tools. For each category, including accommodations, dining, attractions, transportation, and budget, at least one relevant tool should be used during the interaction to gather sufficient information to help the user provide a clear and feasible request.""" + null_tips

ask_prompts["direct"]["user"] = "Conversation: {conversations}\nInteraction trajectory: {trajectories}\nPlease determine whether the user's request needs clarification. A request needs clarification if the user's intention contains missing or unfeasible details based on the tool parameters and call results in the interaction trajectory. If the user's intention requires clarification, answer \"Yes\"; if it is clear and feasible, answer \"No\".\nAnswer:"

ask_prompts["direct"]["ask"] = "Please ask the user one clarification question to gather more information about a specific detail. Do not attempt to solve the task.\nQuestion:"

ask_prompts["proactive"]["system"] = "Your current task is to determine the user's intentions and satisfy their needs based on the provided conversation between the user and the assistant."

ask_prompts["proactive"]["user"] = "Conversation: {conversations}\nBased on the conversation, you have two options: ask a clarifying question or take no action. Choose the appropriate option to formulate your answer, starting with either \"The clarifying question is\" or \"No action should be taken\"." + null_tips

ask_prompts["procot"]["system"] = ask_prompts["proactive"]["system"]

ask_prompts["procot"]["user"] = "Conversation: {conversations}\nBased on the conversation, first determine whether the user's request is ambiguous. A request is ambiguous if it contains missing or unfeasible details. If it is ambiguous, ask a clarifying question. If it is not ambiguous, no action is needed. Your response should start with an analysis of the ambiguity and then conclude with either \"Therefore, the request is not ambiguous. No action should be taken.\" or \"Therefore, the request is ambiguous. The clarifying question is\"." + null_tips

eval_prompts = {}
eval_prompts["system"] = "You are a helpful assistant skilled at evaluating questions."
eval_prompts["user_add"] = 'Please check if the following question exclusively asks for {detail}, rather than {others}. Provide a simple "Yes" or "No" answer.\nQuestion: {question}'
eval_prompts["user_replace"] = 'Please check if the question indicates that the initial {detail} is/are unfeasible and requests changes to the {detail}, rather than {others}. Provide a simple "Yes" or "No" answer.\nQuestion: {question}'
