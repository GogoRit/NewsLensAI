from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimilarityAgent:
    def __init__(self):
        self.agent = Agent(
            model=HuggingFaceChat(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=4096,
                api_key=os.getenv("HUGGINGFACE_API_KEY"),  # Load API key from .env
            ),
            markdown=True,
        )

    def run(self, original_text: str, summary_text: str) -> float:
        """
        Calculate the similarity between the original text and the summary using LLaMA-3.
        Returns a numeric similarity score between 0 and 1.
        """
        prompt = f"""
        Compare the original text and the summary. 
        Provide a similarity score between 0 and 1, where 1 means the texts are identical and 0 means they are completely different.
        Your response must contain only the numeric similarity score and nothing else.

        Original Text: {original_text}
        Summary: {summary_text}
        """
        # Calculate similarity using the agent
        response = self.agent.run(prompt)

        # Extract the numeric similarity score from the response
        try:
            # Use regex to find a floating-point number in the response
            similarity_score = float(re.search(r"\d+\.\d+", response.content).group())
            return similarity_score
        except (ValueError, AttributeError):
            # If no numeric score is found, return 0.0 as a fallback
            return 0.0