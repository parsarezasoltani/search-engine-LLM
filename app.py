import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentType
from langchain.agents.initialize import initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --------------------------------------------------------------
# Load environment
# --------------------------------------------------------------
load_dotenv()

# --------------------------------------------------------------
# Tools
# --------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="search")

# --------------------------------------------------------------
# UI
# --------------------------------------------------------------
st.title("🔎 LangChain - Chat with search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Fallback from environment
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("No API key provided.")
    st.stop()

# --------------------------------------------------------------
# Chat init
# --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --------------------------------------------------------------
# Chat input
# --------------------------------------------------------------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
    )

    tools = [search, arxiv, wiki]

    # IMPORTANT: Zero‑shot ReAct expects a *single* input string.
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5,  # FIX iteration limit failures
        verbose=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
