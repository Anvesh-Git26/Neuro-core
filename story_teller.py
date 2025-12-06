import os
import sys
# Fix imports so we can run from anywhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class StoryTeller:
    def __init__(self):
        print("--- Loading Generative Model (Quantized TinyLlama) ---")
        # Define path to model (assuming it's in the 'models' folder next to src)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '..', 'models', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
        
        # Check if model exists, if not, warn user
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            print("Please download it using: wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/")
            self.chain = None
            return

        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={'max_new_tokens': 256, 'temperature': 0.7, 'context_length': 1024}
        )

        template = """
        You are Miko, a friendly robot companion for children.
        Write a short, engaging bedtime story for a 5-year-old child.
        
        Topic: {topic}
        Mood: {mood}
        
        Story:
        """
        self.prompt = PromptTemplate(input_variables=["topic", "mood"], template=template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, topic, mood):
        if not self.chain:
            return "Error: Model file missing."
        print(f"✨ Generating story about {topic}...")
        return self.chain.invoke({"topic": topic, "mood": mood})

if __name__ == "__main__":
    bot = StoryTeller()
    if bot.chain:
        print(bot.generate("a brave puppy", "Adventurous"))