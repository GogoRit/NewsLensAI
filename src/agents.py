from crewai import Agent, Task
from langchain.chat_models import ChatOpenAI
from src.summarization import generate_summary
from src.hallucination import check_hallucination
from src.bias_analysis import analyze_bias

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load OpenAI Model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Define Crew AI Agents
summary_agent = Agent(
    role="Summarization Expert",
    goal="Generate an accurate summary of the given news article.",
    backstory="A language model specializing in summarization.",
    tools=[generate_summary],
    llm=llm
)

hallucination_agent = Agent(
    role="Fact-Checker",
    goal="Identify hallucinations in the summary by comparing it to the original article.",
    backstory="A language model trained for fact-checking.",
    tools=[check_hallucination],
    llm=llm
)

bias_agent = Agent(
    role="Bias Analyst",
    goal="Analyze the summary and detect potential biases.",
    backstory="A political science expert trained in bias detection.",
    tools=[analyze_bias],
    llm=llm
)

# Define tasks for each agent
summary_task = Task(
    description="Generate a summary from the given article.",
    agent=summary_agent
)

hallucination_task = Task(
    description="Detect hallucinations in the summary.",
    agent=hallucination_agent
)

bias_task = Task(
    description="Perform bias analysis on the summary.",
    agent=bias_agent
)