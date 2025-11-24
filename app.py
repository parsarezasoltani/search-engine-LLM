# Updated and Improved Streamlit + LangChain + Groq Chatbot
# --------------------------------------------------------------
# Improvements:
# - Replaced deprecated initialize_agent + ZERO_SHOT_REACT_DESCRIPTION
#   with the newer AgentExecutor + create_react_agent (more stable).
# - Removed iteration/time-limit issues by adding max_iterations.
# - Updated Groq model name to latest stable.
# - Improved error handling.
# - General code cleanup.

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StreamlitCallbackHandler

# Load local `.env`
load_dotenv()

# --------------------------------------------------------------
# 1️⃣ TOOLS
# --------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = DuckDuckGoSearchRun(name="search")

tools = [search_tool, arxiv_tool, wiki_tool]

# --------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------
st.title("🔎 LangChain – Chat with Search (Groq)")
st.caption("Improved version with ReAct agent + iteration fix.")

# Sidebar -- API Key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password", placeholder="sk-..."
)

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("🚨 No Groq API key found. Add it in the sidebar or set GROQ_API_KEY.")
    st.stop()

# --------------------------------------------------------------
# Chat history
# --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --------------------------------------------------------------
# NEW MESSAGE HANDLER
# --------------------------------------------------------------
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",  # latest Groq Llama update
        streaming=True,
    )

    # Create ReAct agent
    agent = create_react_agent(llm, tools)

    search_agent = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,       # 🔧 Fix iteration runaway
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = search_agent.invoke({"input": prompt}, callbacks=[st_cb])
            final_text = response.get("output", "(No response generated.)")

        except Exception as e:
            final_text = f"⚠️ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": final_text})
        st.write(final_text)
