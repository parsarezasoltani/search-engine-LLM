# --------------------------------------------------------------
# 🚀 Streamlit + LangChain + Groq demo (search, arXiv, Wikipedia)
# --------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports ------------------------------------------------
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# --------------------------------------------------------------
# Load .env (optional – only useful locally)
# --------------------------------------------------------------
load_dotenv()    # allows a local .env file

# --------------------------------------------------------------
# 1️⃣ Tool definitions (WITH CLEARER NAMES)
# --------------------------------------------------------------
# 1️⃣ Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# Give the tool a descriptive name for the agent's internal monologue
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper, name="Arxiv Search") 

# 2️⃣ Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# Give the tool a descriptive name for the agent's internal monologue
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper, name="Wikipedia Search") 

# 3️⃣ DuckDuckGo web‑search tool
search_tool = DuckDuckGoSearchRun(name="web_search")    # Renamed for clarity

tools = [search_tool, arxiv_tool, wiki_tool]

# --------------------------------------------------------------
# 2️⃣ Streamlit UI
# --------------------------------------------------------------
st.title("🔎 LangChain – Chat with search (Groq)")
st.caption(
    "Powered by Groq + LangChain agents. "
    "Enter your Groq API key in the sidebar."
)

# ------------------------------------------------------------------
# Sidebar – API key and Model Selector
# ------------------------------------------------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password", placeholder="sk-..."
)

# Fallback to environment variable
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

# Model selector for flexibility
model_name = st.sidebar.selectbox(
    "Select Model:",
    ("llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-instruct-v0.1"),
    index=0
)

if not api_key:
    st.warning(
        "🚨 No Groq API key found. Add it in the sidebar or as a Space secret "
        "`GROQ_API_KEY`."
    )
    st.stop()

# ------------------------------------------------------------------
# Initialise conversation buffer
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I’m a Groq-powered chatbot that can search the web, arXiv, and Wikipedia. How can I help you?",
        }
    ]

# Render the chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------------------------------------------------------
# 3️⃣ Handle a new user prompt
# ------------------------------------------------------------------
if prompt := st.chat_input(placeholder="Ask me anything…"):
    # Show the user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --------------------------------------------------------------
    # LLM with tool support
    # --------------------------------------------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name, # Use the selected model
        streaming=True,
        tool_choice="auto",
    )

    # --------------------------------------------------------------
    # Build the agent (FIXED: Increased Iteration Limit)
    # --------------------------------------------------------------
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
        verbose=True,
        # 👇 FIX: Increased limits to prevent the "Agent stopped" error
        max_iterations=15, 
        max_execution_time=60, # Generous 60-second limit
    )

    # --------------------------------------------------------------
    # Run the agent and display the answer
    # --------------------------------------------------------------
    with st.chat_message("assistant"):
        # StreamlitCallbackHandler streams the agent's internal “thoughts”
        # Set expand_new_thoughts=True for better transparency
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

        # NOTE: The Zero‑Shot ReAct agent expects a *single* user query.
        response = search_agent.run(prompt, callbacks=[st_cb])

        # Store & render the assistant's final answer
        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
