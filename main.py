from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.chains import ConversationChain
import logging
import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import anthropic
from langchain import PromptTemplate
from langchain.cache import SQLiteCache
import langchain
import cProfile
import pstats
import io

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

class Config:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

config = Config()
anthropic.api_key = config.anthropic_api_key

logging.basicConfig(level=logging.INFO)

# Initialize the memory
memory = ConversationBufferMemory()

class ChainLoader:
    def __init__(self, chat_model_class, conversation_chain_class, callback_manager):
        self.chat_model_class = chat_model_class
        self.conversation_chain_class = conversation_chain_class
        self.callback_manager = callback_manager

    def load_chain(self, model_name="claude-2.0", temperature=0.3, max_tokens_to_sample=75000, streaming=True, verbose=True):
        chat_model = self._initialize_chat_model(model_name, temperature, max_tokens_to_sample, streaming, verbose)
        conversation_chain = self._initialize_conversation_chain(chat_model)
        return conversation_chain

    def _initialize_chat_model(self, model_name, temperature, max_tokens_to_sample, streaming, verbose):
        return self.chat_model_class(
            model=model_name,
            temperature=temperature,
            max_tokens_to_sample=max_tokens_to_sample,
            streaming=streaming,
            verbose=verbose,
            callback_manager=self.callback_manager,
        )

    def _initialize_conversation_chain(self, chat_model):
        return self.conversation_chain_class(
            llm=chat_model,
            memory=memory,
        )

chain_loader = ChainLoader(ChatAnthropic, ConversationChain, CallbackManager([StreamingStdOutCallbackHandler()]))
chain = chain_loader.load_chain()

def read_prompt_template():
    with open('Anthropic_Prompt.txt', 'r') as file:
        return file.read()

template = read_prompt_template()
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

class StreamlitUI:
    def __init__(self, chain, prompt):
        self.chain = chain
        self.prompt = prompt

    def run(self):
        st.set_page_config(page_title="VNTANA Sales", page_icon=":robot:")
        st.header("VNTANA Sales")

        st.session_state.setdefault("generated", [])
        st.session_state.setdefault("past", [])

        user_input = self.get_text()

        if user_input:
            prompt_text = self.prompt.format(history=st.session_state["past"], input=user_input)
            pr = cProfile.Profile()
            pr.enable()
            output = self.chain.run(input=prompt_text)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                st.write(f"AI: {st.session_state['generated'][i]}")
                st.write(f"Human: {st.session_state['past'][i]}")

    def get_text(self):
        return st.text_input("You: ", "", key="input")

ui = StreamlitUI(chain, prompt)
ui.run()
