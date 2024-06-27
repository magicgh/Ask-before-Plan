import os, sys, evaluate
from typing import Dict
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.openai import OpenAIChatEngine
from common.prompts.ask import eval_prompts
from common.chat import extract_binary_answer

detail_in_prompt = {
    "org": "departure city",
    "dest": "destinations and arrival days",
    "days": "trip duration in days",
    "departure_date": "departure date",
    "people_number": "number of people on the trip",
    "housing": "accommodation preferences",
    "cuisine": "cuisine preferences",
    "transportation": "transportation preferences",
    "budget": "budget"
}

bleu = evaluate.load("bleu")
rouge  = evaluate.load("rouge")
    
def rule_based_eval(question, attribute):

    question = question.lower()
    keyword_map = {
        "org": (["origin", "depart", "leave", "start"] if not ("when" in question or "what time" in question) else []),
        "dest": ["destination", "arrive", "reach", "visit"],
        "departure_date": ["date"] + (["depart", "leave", "start"] if ("when" in question or "what time" in question) else []),
        "days": ["days", "duration"],
        "people_number": ["people", "group", "person", "individual", "guest"],
        "budget": ["budget", "money", "afford", "cost", "how much", "expenditure", "pay"],
        "transportation": ["transport", "vehicle", "drive", "fly", "car", "plane", "driving", "flight", "taxi"],
        "housing": ["housing", "hotel", "room", "accommodation"],
        "cuisine": ["food", "cuisine", "eat", "restaurant", "meal"]
    }

    candidates = set()

    for category, key_words in keyword_map.items():
        for key_word in key_words:
            if key_word in question:
                candidates.add(category)
                break
    
    return len(candidates) == 1 and candidates.pop() == attribute

def gpt_evaluation(question, candidate):
    chat_engine = OpenAIChatEngine(model="gpt-4-turbo")
    user_prompt = eval_prompts.get(f"user_{candidate['operation'].lower()}", None)
    assert user_prompt is not None, f"Invalid attribute: {candidate['operation']}"
    other_details = ", ".join([value for key, value in detail_in_prompt.items() if key != candidate["attribute"]])
    if candidate["operation"] == "REPLACE" and candidate["attribute"] == "dest":
        user_prompt = user_prompt.format(detail="destinations", others=other_details, question=question)
    else:
        user_prompt = user_prompt.format(detail=detail_in_prompt[candidate["attribute"]], others=other_details, question=question)
    messages = [
        {
            "role": "system",
            "content": eval_prompts["system"]
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    response, _ = chat_engine.generate(prompt = messages, max_new_tokens=32, temperature=0.0, top_p=1.0)
    return extract_binary_answer(response[0])

def compute_similarity(question: str, candidate: Dict) -> Dict:
    calc_args = {"predictions": [question], "references": [candidate["question"]]}
    scores = {
        "bleu": bleu.compute(**calc_args)['bleu'],
        "rouge": rouge.compute(**calc_args),
    }
    return {
        "bleu": scores['bleu'],
        "rouge1": scores['rouge']['rouge1'],
        "rouge2": scores['rouge']['rouge2'],
        "rougeL": scores['rouge']['rougeL']
    }
    
def fetch_gpt_cache(logs, model_name, prompt_method, data_idx, detail_idx):
    return logs.get(f"{model_name}_{prompt_method}", {}).get("gpt_score", {}).get(str(data_idx), {}).get(str(detail_idx), None)