from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BiasDetectionAgent:
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
        Detect bias in the input text using LLaMA-3.
        """
        prompt = f"""
        Analyze the tone and perspective of the given text and classify its bias as 'Left', 'Right', or 'Center'. 
        Provide a brief explanation for your classification.

        Text: {text}
        """
        # Detect bias using the agent
        response = self.agent.run(prompt)
        return response.content