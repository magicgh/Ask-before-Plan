import os, time, sys
import logging
import openai
import tiktoken
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import importlib
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from transformers import AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI


current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_file_dir, "..")))
sys.path.append(os.path.abspath(os.path.join(current_file_dir, "../tools")))

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.APIConnectionError:
        logging.error("APIConnectionError")
    elif error == openai.RateLimitError:
        logging.error("RateLimitError")
        time.sleep(60)
    elif error == openai.APIError:
        logging.error("APIError")
    elif error == openai.AuthenticationError:
        logging.error("AuthenticationError")
    else:
        logging.error(f"Erorr: {error}")


def init_model(
    model_name: str,
    max_tokens: int,
    temperature: float,
    stop_words: Optional[List[str]] = None,
    port: Optional[int] = None,
) -> ChatOpenAI:
    if model_name.startswith("gpt-"):
        # BUG: openai API error, do not accept 4 stop words
        """if len(stop_words) > 3:
            if "\n" in stop_words:
                stop_words = [stop_word for stop_word in stop_words if stop_word != "\n"]
            else:
                raise ValueError("The number of stop words should be less than 4.")"""
        
        return ChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
            model_kwargs={"stop": stop_words},
        )
    elif model_name.startswith("llama-3"):
        stop_words = ["<|end_of_text|>", "<|eot_id|>"] + (stop_words or [])
        openai_api_base = f"http://localhost:{port}/v1" if port else "http://localhost:10086/v1"
        return ChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key="EMPTY",
            openai_api_base=openai_api_base,
            model_name=model_name,
            model_kwargs={"stop": stop_words},
        )
    elif model_name.startswith("mistral"):
        stop_words = ["</s>"] + (stop_words or [])
        openai_api_base = f"http://localhost:{port}/v1" if port else "http://localhost:10087/v1"
        return ChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key="EMPTY",
            openai_api_base=openai_api_base,
            model_name=model_name,
            model_kwargs={"stop": stop_words},
        )
    elif model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-1.5-flash-latest",
            max_output_tokens=max_tokens,
        )
    else:
        raise ValueError("Invalid model name {}".format(model_name))


def init_tokenizer(model_name: str):
    if model_name.startswith("gpt-3.5"):
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif model_name.startswith("gpt-4"):
        return tiktoken.encoding_for_model("gpt-4-turbo")
    elif model_name.startswith("llama-3"):
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif model_name.startswith("mistral"):
        return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name.startswith("gemini"):
        return None
    else:
        raise ValueError("Invalid model name {}".format(model_name))


def check_max_input_tokens(model_name: str, max_input_tokens: int, max_new_tokens: int) -> int:
    model_name = model_name.lower()
    if model_name.startswith("gpt-3.5"):
        return min(max_input_tokens, 16000 - max_new_tokens)
    elif model_name.startswith("gpt-4"):
        return min(max_input_tokens, 120000 - max_new_tokens)
    elif model_name.startswith("llama-3"):
        return min(max_input_tokens, 8000 - max_new_tokens)
    elif model_name.startswith("mistral"):
        return min(max_input_tokens, 32000 - max_new_tokens)
    elif model_name.startswith("gemini"):
        return min(max_input_tokens, 1000000 - max_new_tokens)
    else:
        return max_input_tokens


def format_text(text: str) -> str:
    return text.strip()

def truncate_text(text: str, stop_words: List[str]) -> str:
    for stop_word in stop_words:
        text = text.split(stop_word)[0]
    return text

class Agent:
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
        max_steps: int,
        prompt_lib: Dict,
        stop_words: Optional[List[str]] = None,
        port: Optional[int] = None,
    ):
        self.model_name = model_name
        self.model = init_model(model_name, max_new_tokens, temperature, stop_words, port)
        self.max_input_tokens = check_max_input_tokens(model_name, max_input_tokens, max_new_tokens)
        self.max_steps = max_steps
        self.encoder = init_tokenizer(model_name)
        self.prompt_lib = prompt_lib
        self.stop_words = stop_words
        self.messages = []

    def generate(self, custom_messages: List[str] = None) -> str:
        while True:
            try:
                if custom_messages:
                    if self._exceed_max_tokens(custom_messages):
                        logging.warning(f"Your input exceeds the maximum token limit. Current token length: {self._get_message_length(custom_messages)}")
                        
                    if self.model_name.startswith("gemini"):
                        request = format_text(self.model.invoke(custom_messages, stop=self.stop_words).content)
                        """elif self.model_name.startswith("gpt-"):
                            # BUG: openai API error, do not accept 4 stop words
                            request = truncate_text(format_text(self.model.invoke(custom_messages).content), self.stop_words)"""
                    else:
                        request = format_text(self.model.invoke(custom_messages).content)
                else:
                    if self._exceed_max_tokens():
                        logging.warning(f"Your input exceeds the maximum token limit. Current token length: {self._get_message_length(self.messages)}")
                    if self.model_name.startswith("gemini"):
                        request = format_text(self.model.invoke(self.messages, stop=self.stop_words).content)
                        """elif self.model_name.startswith("gpt-"):
                            # BUG: openai API error, do not accept 4 stop words
                            request = truncate_text(format_text(self.model.invoke(self.messages).content), self.stop_words)"""
                    else:
                        request = format_text(self.model.invoke(self.messages).content)
                logging.info("request: {}".format(request))
                return request
            except:
                catch_openai_api_error()
                time.sleep(1)

    def _build_prompt(self, role: str, **kwargs) -> str:
        template = self.prompt_lib[role]
        if kwargs:
            return template.format(**kwargs)
        else:
            return template

    def _build_system_message(self, **kwargs) -> str:
        return [SystemMessage(self._build_system_prompt(**kwargs))]

    def _build_user_message(self, **kwargs) -> str:
        return [HumanMessage(self._build_user_prompt(**kwargs))]

    def _build_agent_message(self, **kwargs) -> str:
        return [AIMessage(self._build_agent_prompt(**kwargs))]

    def _build_agent_prompt(self, **kwargs) -> str:
        return self._build_prompt("assistant", **kwargs)

    def _build_user_prompt(self, **kwargs) -> str:
        return self._build_prompt("user", **kwargs)

    def _build_system_prompt(self, **kwargs) -> str:
        return self._build_prompt("system", **kwargs)

    def _build_custom_prompt(self, role: str, **kwargs) -> str:
        return self._build_prompt(role, **kwargs)

    def _get_token_length(self, text: str) -> int:
        if self.model_name.startswith("gemini"):
            return self.model.get_num_tokens(text)
        else:
            return len(self.encoder.encode(text))
        
    def _get_message_length(self, messages: List[str]) -> int:
        return self._get_token_length("\n".join([message.content for message in messages]))
            

    def _exceed_max_tokens(self, custom_messages: List[str] = None) -> bool:
        if custom_messages:
            input_token_length = self._get_message_length(custom_messages)
        else:
            input_token_length = self._get_message_length(self.messages)
        return input_token_length > self.max_input_tokens

    def reset(self):
        self.messages.clear()


class ReactAgent(Agent):

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
        max_steps: int,
        prompt_lib: Dict,
        env_names: List[str],
        stop_words: List[str] = ["Action", "Thought", "Observation", "\n"],
        port: Optional[int] = None,
    ):
        super().__init__(
            model_name,
            max_input_tokens,
            max_new_tokens,
            temperature,
            max_steps,
            prompt_lib,
            stop_words,
            port
        )

        self.curr_step = 0
        self.envs = self.load_tools(env_names)
        self.scratchpad = ""
        self.done = False

    def load_tools(self, tools: List[str]) -> Dict[str, Any]:
        tools_map = {}
        postprocess_budget = False

        if "budget" in tools:
            dependencies = ["accommodations", "flights", "googleDistanceMatrix"]
            for dependency in dependencies:
                if dependency not in tools:
                    raise ValueError(
                        f"Budget tool requires {dependency} tool to be loaded."
                    )
            tools.remove("budget")
            postprocess_budget = True

        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))

            if tool_name == "calculator":
                if isinstance(self, ReflectAgent):
                    tools_map[tool_name] = getattr(module, "ReflectEnv")()
                elif isinstance(self, ReactAgent):
                    tools_map[tool_name] = getattr(module, "ReactEnv")()
                else:
                    raise ValueError("Invalid agent type for calculator tool")

            else:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )()

        if postprocess_budget:
            budget_module = importlib.import_module("tools.budget.apis")
            tools_map["budget"] = getattr(budget_module, "Budget")(
                tools_map["accommodations"],
                tools_map["flights"],
                tools_map["googleDistanceMatrix"],
            )

        return tools_map

    def run(self, dialogs: List, reset: bool = True) -> None:

        self.messages += dialogs

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_done()):
            self.step()

        return self.scratchpad

    def is_done(self) -> bool:
        return self.done

    def is_halted(self) -> bool:
        return (
            (self.curr_step >= self.max_steps) or (self._exceed_max_tokens())
        ) and not self.done

    def reset(self) -> None:
        super().reset()
        self.scratchpad = ""
        self.curr_step = 0
        self.done = False


class ReflectAgent(ReactAgent):
    """
    A question answering Self-Reflecting React Agent.
    """

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
        max_steps: int,
        prompt_lib: Dict,
        env_names: List[str],
        react_stop_words: List[str] = ["Action", "Thought", "Observation", "\n"],
        reflect_stop_words: List[str] = ["\n"],
        port: Optional[int] = None,
    ):

        super().__init__(
            model_name,
            max_input_tokens,
            max_new_tokens,
            temperature,
            max_steps,
            prompt_lib,
            env_names,
            react_stop_words,
            port
        )
        self.rationales = []
        assert len(reflect_stop_words) <= 3
        self.reflect_model = init_model(model_name, max_new_tokens, temperature, reflect_stop_words, port)
        self.reflect_stop_words = reflect_stop_words

    def format_rationales(self):
        if self.rationales == []:
            return ""
        else:
            return (
                self.prompt_lib["header"]
                + "\nReflections:\n- "
                + "\n- ".join([r.strip() for r in self.rationales])
            )

    def think(self, reflection_messages: List[str] = None) -> str:
        while True:
            try:
                if self._exceed_max_tokens(reflection_messages):
                    logging.warning(f"Your input exceeds the maximum token limit. Current token length: {self._get_message_length(reflection_messages)}")
                    
                if self.model_name.startswith("gemini"):
                    request = format_text(self.reflect_model.invoke(reflection_messages, stop=self.reflect_stop_words).content)
                else:
                    request = format_text(self.reflect_model.invoke(reflection_messages).content)
                logging.info("request: {}".format(request))
                return request
            except:
                catch_openai_api_error()
                time.sleep(1)
    
    def reflect(self, **kwargs) -> str:
        logging.info("Reflecting...")
        self.rationales += [self.think(self._build_reflection_message(**kwargs))]
        self.rationales = list(set(self.rationales))

    def _build_reflection_message(self, **kwargs) -> str:
        return [HumanMessage(self._build_reflection_prompt(**kwargs))]

    def _build_reflection_prompt(self, **kwargs) -> str:
        return self._build_prompt("reflection", **kwargs)

    def reset(self) -> None:
        super().reset()
        self.rationales.clear()
