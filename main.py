from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="nvidia/nemotron-3-nano-30b-a3b:free",
                 temperature=2, 
                   openai_api_key=os.getenv("OPENAI_API_KEY"), 
                   openai_api_base="https://openrouter.ai/api/v1")


chat_history = [SystemMessage(content= "you are a helpful assistant")]

while True:
    user_input = input("User:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)


print(chat_history)



