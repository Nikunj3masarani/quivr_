import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage

os.environ['AZURE_OPENAI_API_KEY'] = 'a82b2d9fd7e94efbad345ba4b19e6b16'
os.environ['OPENAI_API_VERSION'] = '2023-12-01-preview'
os.environ['AZURE_API_BASE'] = 'https://globalip-sm.openai.azure.com/'

chat = ChatLiteLLM(
    model="azure/gpt-4-turbo"
)

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
print(chat(messages))
