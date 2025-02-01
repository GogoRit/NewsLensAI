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

hallucination_prompt = PromptTemplate(
    input_variables=["article", "summary"],
    template="Compare the summary to the original article. Identify any hallucinated (false or misleading) facts:\n\nArticle: {article}\nSummary: {summary}\n\nHallucination Check:"
)

def check_hallucination(article, summary):
    chain = LLMChain(llm=llm, prompt=hallucination_prompt)
    return chain.run(article=article, summary=summary)