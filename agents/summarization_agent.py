from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SummarizationAgent:
    def __init__(self):
        self.agent = Agent(
            model=HuggingFaceChat(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=4096,
                api_key=os.getenv("HUGGINGFACE_API_KEY"),  # Load API key from .env
            ),
            markdown=True,
        )

    def run(self, text: str) -> str:
        """
        Summarize the input text using LLaMA-3.
        """
        prompt = f"""
        Can you provide a comprehensive summary in maximum 50 words of the given text? 
        The summary should cover all the key points and main ideas presented in the original text, 
        condensing the information into a concise and easy-to-understand format. 
        Please ensure that the summary includes relevant details and examples that support the main ideas, 
        while avoiding unnecessary information or repetition. 
        The summary should be appropriately condensed without omitting any important information.

        Text: {text}
        """
        # Generate summary using the agent
        response = self.agent.run(prompt)
        return response.content