from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Read the prompt template from a .txt file
with open('/Users/benconway/Documents/GitHub/VNTANA-sales-chatbot/Anthropic_Prompt.txt', 'r') as file:
    template = file.read()

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

# Initialize the memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the LLMChain with the ChatAnthropic model
llm_chain = LLMChain(
    llm=ChatAnthropic(
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Interactive chat
while True:
    human_input = input("You: ")
    if human_input.lower() == "quit":
        break
    response = llm_chain.predict(human_input=human_input)
    print("AI: " + response)
