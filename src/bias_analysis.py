from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

bias_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Analyze the following news summary for bias. Identify any partisan framing or misleading information:\n\nSummary: {summary}\n\nBias Analysis:"
)

def analyze_bias(summary):
    chain = LLMChain(llm=llm, prompt=bias_prompt)
    return chain.run(summary)