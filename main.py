from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.chains import ConversationChain
import logging
import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)

anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')

# Read the prompt template 
with open('Anthropic_Prompt.txt', 'r') as file:
  template = file.read()

# Use 'history' and 'input' as per the ConversationChain requirements
prompt = PromptTemplate(
  input_variables=["history", "input"],
  template=template
)

# Initialize the memory
memory = ConversationBufferMemory()

def load_chain(model_name="claude-2.0", temperature=0.3, max_tokens_to_sample=75000, streaming=True, verbose=True):
  # Initialize the LLM
  llm = ChatAnthropic(
     model=model_name,
     temperature=temperature,
     max_tokens_to_sample=max_tokens_to_sample,
     streaming=True,
     verbose=verbose,
     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
  )

  # Initialize the ConversationChain
  conversation = ConversationChain(
    llm=llm,
    memory=memory # pass the memory here
  )

  return conversation

chain = load_chain()

# Streamlit UI
st.set_page_config(page_title="VNTANA Sales", page_icon=":robot:")
st.header("VNTANA Sales")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "How can I help you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    # Use the prompt template to generate the prompt
    prompt_text = prompt.format(history=st.session_state["past"], input=user_input)
    output = chain.run(input=prompt_text)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.write(f"AI: {st.session_state['generated'][i]}")
        st.write(f"Human: {st.session_state['past'][i]}")
