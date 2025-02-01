from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load OpenAI Model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Define the prompt for summarization
summary_prompt = PromptTemplate(
    input_variables=["article"],
    template="Summarize the following news article:\n\n{article}"
)

# Create LangChain summarization pipeline
def generate_summary(article):
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    return chain.run(article)