import streamlit as st
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Hugging Face API client with token from environment
client = InferenceClient(
    "microsoft/Phi-3.5-mini-instruct",
    token=os.getenv("HF_TOKEN")  # Load safely from .env
)

# System prompt for hackathon idea evaluation
SYSTEM_PROMPT_EVALUATOR = """
Role: Hackathon Judge

Instructions: You are an expert judge responsible for evaluating hackathon ideas. Your evaluation should consider the following aspects, ensuring the total length does not exceed 800 tokens:

1. **Input Assessment**: First, assess if the user’s input is related to hackathon ideas or project development. 
   - If the input is not relevant, respond with: "The provided input does not pertain to hackathon ideas or project development. Please ask about hackathon ideas or evaluation only." without additional commentary.

2. If the input is relevant, proceed with the evaluation:
   - **Originality** (score): Provide a few lines explaining how unique the idea is and whether it reflects a new vision or approach to a problem.
   - **Feasibility** (score): Analyze if the project can realistically be developed within the hackathon timeframe and resources.
   - **Effectiveness** (score): Evaluate how well the solution addresses the specified problem and its potential to make a real-world impact.
   - **Scalability** (score): Determine the project's potential for growth and application across various contexts or demographics.
   - **Tech Stack Fit**: List the essential technologies and tools needed for the project:
     - Programming languages (e.g., Python, JavaScript)
     - Frameworks (e.g., Flask, React)
     - Databases (e.g., MongoDB, PostgreSQL)
     - APIs (e.g., third-party services relevant to the project)

3. Finally, provide an overall score out of 10 for the idea and evaluate the chances of winning based on its originality, feasibility, effectiveness, and scalability.

Ensure the total response is concise and does not exceed 800 tokens. 
Make sure examples from previous projects are **accessible, simple, and relatable** for a hackathon setting.

Response Style:
- Provide a concise yet thorough evaluation.
- Offer short, effective feedback to improve the idea, if necessary.
- Include **realistic and relatable examples** from previous successful projects, but avoid highly advanced examples unless the project warrants it.
"""

# Template for the hackathon idea
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input"],
    template="{system_prompt}\n\nUser: {user_input}\nAssistant:"
)

# Streamlit app layout
st.title("Hackathon Idea Evaluator")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome! Enter your hackathon idea, and I'll help you evaluate it."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to assess if input is hackathon-related
def is_relevant_input(user_input):
    keywords = ["idea", "project", "evaluate", "hackathon", "build", "develop", "solution", "problem", "AI", "machine learning", "code", "technology"]
    if any(keyword in user_input.lower() for keyword in keywords):
        return True
    if len(user_input.split()) > 3 and "?" not in user_input:
        return True
    return False

# Chat input and processing
if idea_input := st.chat_input("Enter your hackathon idea for evaluation:"):
    st.session_state.messages.append({"role": "user", "content": idea_input})
    st.chat_message("user").write(idea_input)

    if not is_relevant_input(idea_input):
        response = "The provided input does not pertain to hackathon ideas or project development. Please ask about hackathon ideas or evaluation only."
    else:
        formatted_prompt = prompt_template.format(
            system_prompt=SYSTEM_PROMPT_EVALUATOR,
            user_input=idea_input
        )

        response = ""
        try:
            for message in client.chat_completion(
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=800,
                stream=True,
            ):
                response += message.choices[0].delta.content

            if "tokens" in response:
                response = response.split("(total tokens:")[0].strip()
            if "Assistant:" in response:
                response = response.split("Assistant:")[1].strip()

        except Exception as e:
            response = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
    st.chat_message("assistant").write(response.strip())
