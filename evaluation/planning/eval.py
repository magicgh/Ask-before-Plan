import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from commonsense_constraint import evaluation as commonsense_eval
from hard_constraint import evaluation as hard_eval
from tqdm import tqdm
import argparse
from common.data import load_data, extract_results, expand_task
from common.eval import fetch_eval_files, load_evals, write_evals, open_eval_file
from filelock import FileLock


def combine_json_data(eval_files, model_name, prompt_method):
    sorted_data = sorted(extract_results(eval_files, "plan", model_name, prompt_method), key=lambda x: x[0])
    return [x[1] for x in sorted_data]

def count_true_false(data):
    """Count the number of true and false values in a list."""
    true_count = data.count(True)
    false_count = data.count(False)
    return true_count, false_count

def statistics(commonsense_statistic):
    """Generate statistics for each level and day in the given data with a different structure."""
    result = {level: {day: {} for day in commonsense_statistic[level]} for level in commonsense_statistic}
    
    for level, days in commonsense_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count
                
    return result

def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    mapping_dict = {'is_valid_information_in_current_city':'Within Current City','is_valid_information_in_sandbox':'Within Sandbox','is_reasonalbe_visiting_city':'Reasonable City Route','is_valid_restaurants':'Diverse Restaurants','is_valid_transportation':'Non-conf. Transportation','is_valid_attractions':'Diverse Attractions','is_valid_accommodation':'Minimum Nights Stay','is_not_absent':'Complete Information','valid_cost':'Budget','valid_housing':'housing','valid_cuisine':'Cuisine','valid_transportation':'Transportation'}
    remap_commonsense_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    remap_hard_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense_constraint_record[level][day] = {mapping_dict[key] : val for key,val in commonsense_constraint_record[level][day].items()}
            remap_hard_constraint_record[level][day] = {mapping_dict[key] : val for key,val in hard_constraint_record[level][day].items()}
    return remap_commonsense_constraint_record, remap_hard_constraint_record


def eval_score(data_split: str, eval_dir: str, model_name: str, prompt_method: str):

    query_data_list = load_data(split=data_split)
    
    eval_files = fetch_eval_files(os.path.join(eval_dir, data_split))
    assert len(eval_files) == len(query_data_list)
    
    query_data_list = [list(expand_task(x))[-1][-1] for x in query_data_list]
    hardConstraint_statistic= {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    commonsenseConstraint_statistic = {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    tested_plans = combine_json_data(eval_files, model_name, prompt_method)
    assert len(query_data_list) == len(tested_plans)
    delivery_cnt = 0
    plan_constraint_store = []
    for idx in tqdm(range(0,len(query_data_list))):
        query_data = query_data_list[idx]
        tested_plan = tested_plans[idx]

        if tested_plan:
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data,tested_plan)
        else:
            commonsense_info_box = None

        if commonsense_info_box and commonsense_info_box['is_not_absent'][0] and commonsense_info_box['is_valid_information_in_sandbox'][0]:
            hard_info_box = hard_eval(query_data,tested_plan)
        else:
            hard_info_box = None

        plan_constraint_store.append({'commonsense_constraint':commonsense_info_box,'hard_constraint':hard_info_box})

        commonsenseConstraint_statistic[query_data['level']][query_data['days']].append(commonsense_info_box)
        hardConstraint_statistic[query_data['level']][query_data['days']].append(hard_info_box)

    constraint_record = {key: {day: {'housing':0, 'cuisine':0, 'transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    constraint_mapping = {'housing':'valid_housing','cuisine':'valid_cuisine','transportation':'valid_transportation'}
    mapping_constraint_record = {key: {day: {'valid_housing':0, 'valid_cuisine':0, 'valid_transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    count_record = {key:{day:0 for day in [3,5,7]} for key in ['easy','medium','hard']}

    for unit in query_data_list:
        count_record[unit['level']][unit['days']] += 1
        for key in constraint_record['medium'][3]:
            if len(unit[key]) > 0:
                constraint_record[unit['level']][unit['days']][key] += 1
                mapping_constraint_record[unit['level']][unit['days']][constraint_mapping[key]] += 1
    
    commonsenseConstraint_statistic_processed = statistics(commonsenseConstraint_statistic)
    hardConstraint_statistic_processed = statistics(hardConstraint_statistic)


    data_record = {key:{day:[] for day in [3,5,7]} for key in ['easy','medium','hard']}

    constraint_dis_record = {"commonsense":{"pass":0,"total":0},"hard":{"pass":0,"total":0}}
    constraint_count = {key:{day:{} for day in [3,5,7]} for key in ['easy','medium','hard']}

    for constraint in ['commonsense','hard']:
        if constraint == 'commonsense':
            constraint_statistic = commonsenseConstraint_statistic_processed
        elif constraint == 'hard':
            constraint_statistic = hardConstraint_statistic_processed

        key_dict = {'commonsense':['is_valid_information_in_current_city','is_valid_information_in_sandbox','is_reasonalbe_visiting_city','is_valid_restaurants','is_valid_transportation','is_valid_attractions','is_valid_accommodation','is_not_absent'],'hard':['valid_cost','valid_housing','valid_cuisine','valid_transportation']}
        
        for key in constraint_statistic:
            for key2 in constraint_statistic[key]:
                if key2 == -1:
                    print(constraint_statistic[key])
                    exit(0)
                for key3 in key_dict[constraint]:
                    data_record[key][key2].append('0/0')
                    if key3 in constraint_statistic[key][key2]:
                        constraint_dis_record[constraint]['pass'] += constraint_statistic[key][key2][key3]['true']
                        if constraint == 'hard':
                            if key == 'hard' and key3 in ['valid_housing','valid_cuisine','valid_transportation']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            elif key == 'medium' and key3 in ['valid_housing','valid_cuisine']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            else:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                                if key3 in ['valid_cost','valid_days']:
                                    constraint_dis_record[constraint]['total'] += count_record[key][key2]
                                    constraint_count[key][key2][key3] = count_record[key][key2]
                                    hardConstraint_statistic_processed[key][key2][key3]['total'] = count_record[key][key2]
                        else:
                            data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                            constraint_dis_record[constraint]['total'] += count_record[key][key2]
                            constraint_count[key][key2][key3] = count_record[key][key2]
                            commonsenseConstraint_statistic_processed[key][key2][key3]['total'] =  count_record[key][key2]
    final_all_cnt = 0
    final_commonsense_cnt = 0
    final_hardConstraint_cnt = 0
    final_all_cnt_map = {level:0 for level in ['easy','medium','hard']}
    for idx in (range(0,len(query_data_list))):
        if plan_constraint_store[idx]['commonsense_constraint']:
            final_commonsense_pass = True
            final_hardConstraint_pass = True
            for item in plan_constraint_store[idx]['commonsense_constraint']:
                if plan_constraint_store[idx]['commonsense_constraint'][item][0] is not None and not plan_constraint_store[idx]['commonsense_constraint'][item][0]:
                    final_commonsense_pass = False
                    break
            if plan_constraint_store[idx]['hard_constraint'] is None:
                continue
            for item in plan_constraint_store[idx]['hard_constraint']:
                if plan_constraint_store[idx]['hard_constraint'][item][0] is not None and  plan_constraint_store[idx]['hard_constraint'][item][0] == False:
                    final_hardConstraint_pass = False
                    break
                
            if final_commonsense_pass:
                final_commonsense_cnt += 1
            if final_hardConstraint_pass:
                final_hardConstraint_cnt += 1
            if final_commonsense_pass and final_hardConstraint_pass:
                final_all_cnt += 1
                final_all_cnt_map[query_data_list[idx]['level']] += 1

    result = {}

    remap_commonsense_constraint_record, remap_hard_constraint_record = paper_term_mapping(commonsenseConstraint_statistic_processed, hardConstraint_statistic_processed)

    if data_split == 'train':
        assert constraint_dis_record['commonsense']['total'] == 8000
        result['Delivery Rate'] = delivery_cnt / 1000
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / constraint_dis_record['commonsense']['total']
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 1000
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / constraint_dis_record['hard']['total']
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 1000
        result['Final Pass Rate'] = final_all_cnt / 1000

    elif data_split == 'test':
        assert constraint_dis_record['commonsense']['total'] == 8000
        result['Delivery Rate'] = delivery_cnt / 1000
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / constraint_dis_record['commonsense']['total']
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 1000
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / constraint_dis_record['hard']['total']
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 1000
        result['Final Pass Rate'] = final_all_cnt / 1000
    

    return result, {"Commonsense Constraint":remap_commonsense_constraint_record, "Hard Constraint":remap_hard_constraint_record}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--evaluation_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="greedy")
    parser.add_argument("--prompt_method", type=str, default="search")
    args = parser.parse_args()

    save_path = os.path.join(args.evaluation_dir, args.data_split)
    final_results_path, final_logs_path = load_evals(save_path, "plan")
    scores, detailed_scores = eval_score(args.data_split, args.evaluation_dir, args.model_name, args.prompt_method)
    
    with FileLock(final_results_path + ".lock"):
        with FileLock(final_logs_path + ".lock"):
            final_results = open_eval_file(final_results_path, "plan")
            final_details = open_eval_file(final_logs_path, "plan")
            final_results["plan"][f"{args.model_name}_{args.prompt_method}"] = scores
            final_details["plan"][f"{args.model_name}_{args.prompt_method}"] = detailed_scores
            write_evals(save_path, final_results, final_details)
        
        