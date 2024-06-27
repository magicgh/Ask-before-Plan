import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from typing import List
from src.execution.utils import params_regex
from common.chat import parse_tool

def main_eval(candidates: List[str], labels: List[str]):
    if not candidates and not labels:
        return {
            "api_match": 1,
            "precision": 1,
            "recall": 1,
            "f1": 1,
            "repeat_rate": 0,
            "pass_rate": 1
        }
    candidates_set = set([candidate.replace('\'', '"') for candidate in candidates])
    labels_set = set([label.replace('\'', '"') for label in labels])
    candidate_api_set, label_api_set = set(), set()
    
    for candidate in candidates_set:
        action_type, _ = parse_tool(candidate)
        if action_type is not None:
            candidate_api_set.add(action_type)
            
    for label in labels_set:
        action_type, _ = parse_tool(label)
        assert action_type is not None, f"Label {label} is not well-formed"
        label_api_set.add(action_type)
    
    if not candidates:
        return {
            "api_match": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "repeat_rate": 0,
            "pass_rate": 0
        }
    elif not labels:
        return {
            "api_match": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "repeat_rate": 1 - len(candidates_set)/len(candidates),
            "pass_rate": 0
        }
        
    else:
        return {
        "api_match": len(candidate_api_set.intersection(label_api_set))/len(label_api_set),
        "precision": len(candidates_set.intersection(labels_set))/len(candidates_set),
        "recall": len(candidates_set.intersection(labels_set))/len(labels_set),
        "f1": 2 * len(candidates_set.intersection(labels_set))/(len(labels_set) + len(candidates_set)),
        "repeat_rate": 1 - len(candidates_set)/len(candidates),
        "pass_rate": int(labels_set == candidates_set)
    }
    
def well_formed_eval(candidates: List[str]):
    if not candidates:
        return 1
    well_formed_count = 0
    for candidate in candidates:
        action_type, params = parse_tool(candidate)
        if params_regex.get(action_type, None) is not None:
            if re.match(params_regex[action_type], params):
                well_formed_count += 1
    return well_formed_count/len(candidates)
                
        