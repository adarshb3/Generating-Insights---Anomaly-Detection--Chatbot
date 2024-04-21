import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure the OpenAI API key is loaded from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client with the API key
openai_client = OpenAI(api_key=OPENAI_API_KEY, temperature=0)

# Streamlit page configuration
st.set_page_config(page_title="CSV Agent Chatbot", page_icon="🤖", layout='wide')

# Streamlit UI
st.title("Generating Insights - Anomaly Detection- Chatbot")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Save uploaded file
    with open("temp_uploaded.csv", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("File uploaded successfully.")

    # Create the CSV agent
    agent = create_csv_agent(
        openai_client,
        "temp_uploaded.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # User input for queries
    query = st.text_input("Enter your query:")

    if st.button("Get Insights"):
        if query:
            try:
                response = agent.run(query)
                st.write("Response:", response)
            except Exception as e:
                st.error("Error processing the query: " + str(e))
        else:
            st.warning("Please enter a query to get insights.")

