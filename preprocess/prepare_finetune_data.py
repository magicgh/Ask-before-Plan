import os, sys, json, argparse, logging, tiktoken
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dialogue import generate_trajectories
from common.data import load_data, expand_task, generate_message
from common.prompts.tool import tool_prompts
from common.prompts.ask import ask_prompts

logging.basicConfig(level=logging.INFO)
def create_tool_dialog(conversation, trajectories):
    return [
        {
            "role": "system",
            "content": tool_prompts["direct"]["system"]
        },
        {
            "role": "user",
            "content": tool_prompts["direct"]["user"].format(conversations=conversation)
        },
        {
            "role": "assistant",
            "content": "\n".join(trajectories)
        }
    ]

def create_ask_dialog(conversation, trajectories, next_question):
    if trajectories is not None:
        draft_conversation = [
            {"role": "system", "content": ask_prompts["direct"]["system"]},
            {
                "role": "user",
                "content": ask_prompts["direct"]["user"].format(
                    conversations=conversation, trajectories=trajectories
                ),
            },
        ]
    else:
        draft_conversation = [
            {"role": "system", "content": ask_prompts["conversation_only"]["system"]},
            {
                "role": "user",
                "content": ask_prompts["conversation_only"]["user"].format(
                    conversations=conversation
                ),
            },
        ]
    if next_question is not None:
        draft_conversation += [
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": ask_prompts["direct"]["ask"] if trajectories is not None else ask_prompts["conversation_only"]["ask"]},
            {"role": "assistant", "content": next_question},
        ]
    
    else:
        draft_conversation += [{"role": "assistant", "content": "No."}]
        
    return draft_conversation

def filter_conversation_turns(conversation_turns):
    return [turn for turn in conversation_turns if turn[0] is None or turn[0]["operation"] != "REPLACE"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--task_name", type=str, default="tool", choices=["tool", "ask"])
    parser.add_argument("--output_dir", type=str, default="./sft_data")
    parser.add_argument("--conversation_only", action="store_true", default=False)
    
    args = parser.parse_args()
    assert args.conversation_only is True and args.task_name == "ask", "Conversation only mode is only available for the ask task."
    
    cleaned_data = load_data(split=args.data_split)
    
    save_path = os.path.join(args.output_dir, f"{args.task_name}_{args.data_split}.json")
    if args.conversation_only:
        save_path = save_path.replace(".json", "_conversation_only.json")
    
    finetune_data = []
    
    if args.task_name == "tool":
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            messages = [generate_message("user", sample["query"])]
            for modified, current in expand_task(sample):
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))
                    
                trajectories = generate_trajectories(current, output_format="finetuning")
                finetune_data.append({"task": data_idx, "messages": create_tool_dialog(messages, trajectories)})

    else:
        for data_idx, sample in enumerate(tqdm(cleaned_data)):
            messages = [generate_message("user", sample["query"])]
            conversation_turns = list(expand_task(sample))
            if args.conversation_only:
                conversation_turns = filter_conversation_turns(conversation_turns)
            for turn_idx, (modified, current) in enumerate(conversation_turns):
                if modified is not None:
                    messages.append(generate_message("assistant", modified["question"]))
                    messages.append(generate_message("user", modified["answer"]))
                
                if args.conversation_only:
                    trajectories = None
                else:
                    trajectories = generate_trajectories(current, output_format="clarification")
                next_question = None if (turn_idx == len(conversation_turns) - 1 or conversation_turns[turn_idx + 1][0]["operation"] == "REPLACE") else conversation_turns[turn_idx + 1][0]["question"]
                finetune_data.append({"task": data_idx, "messages": create_ask_dialog(messages, trajectories, next_question)})

    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    logging.info("Max token length: {}".format(max([len(encoder.encode(json.dumps(entry["messages"]))) for entry in finetune_data])))
    with open(save_path, "w") as f:
        json.dump(finetune_data, f)
