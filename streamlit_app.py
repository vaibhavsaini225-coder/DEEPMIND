import streamlit as st
from app import create_agent, chat  # USE app.py

st.set_page_config(page_title="LangChain Agent", layout="wide")
st.title("LangChain Groq Agent with Tools")

@st.cache_resource
def init_agent():
    return create_agent()

agent_executor = init_agent()

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your message...")

if user_input:
    response, st.session_state.history = chat(
        user_input,
        agent_executor,
        st.session_state.history
    )

for role, message in st.session_state.history:
    st.markdown(f"**{role}:** {message}")
