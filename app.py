# --------------------------------------------------------------
#  Streamlit + LangChain + Groq demo (search, arXiv, Wikipedia)
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
#  Load .env (optional – only useful locally)
# --------------------------------------------------------------
load_dotenv()   # allows a local .env file; on HF Spaces the secret is read from env vars

# --------------------------------------------------------------
#  1️⃣  Tool definitions
# --------------------------------------------------------------
# 1️⃣ Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# 2️⃣ Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# 3️⃣ DuckDuckGo web‑search tool
search_tool = DuckDuckGoSearchRun(name="search")   # name must be lowercase if you ever pin it

tools = [search_tool, arxiv_tool, wiki_tool]

# --------------------------------------------------------------
# 2️⃣  Streamlit UI
# --------------------------------------------------------------
st.title("🔎 LangChain – Chat with search (Groq)")
st.caption(
    "Powered by Groq + LangChain agents. "
    "Enter your Groq API key in the sidebar (or store it as a Space secret)."
)

# ------------------------------------------------------------------
# Sidebar – API key input (or read from HF secret)
# ------------------------------------------------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password", placeholder="sk-..."
)

# If the key is not typed in the sidebar, try to read it from the
# environment – this works when you add a secret called `GROQ_API_KEY`
# in the Space settings.
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

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
            "content": "Hi, I’m a chatbot that can search the web. How can I help you?",
        }
    ]

# Render the chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------------------------------------------------------
# 3️⃣  Handle a new user prompt
# ------------------------------------------------------------------
if prompt := st.chat_input(placeholder="Ask me anything…"):
    # Show the user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --------------------------------------------------------------
    # LLM with tool support (the only line that changed)
    # --------------------------------------------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama3-8b-8192",          # you can also use "llama3-8b-8192", etc.
        streaming=True,
        # 👇 THIS ENABLES FUNCTION‑CALLING / TOOL USAGE
        tool_choice="auto",
    )

    # --------------------------------------------------------------
    # Build the agent
    # --------------------------------------------------------------
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
        verbose=True,          # prints LangChain internal logs to the Space console
    )

    # --------------------------------------------------------------
    # Run the agent and display the answer
    # --------------------------------------------------------------
    with st.chat_message("assistant"):
        # StreamlitCallbackHandler streams the agent's internal “thoughts”
        # into the same container that the assistant message lives in.
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # NOTE: The Zero‑Shot ReAct agent expects a *single* user query,
        # not the whole list of previous messages.
        response = search_agent.run(prompt, callbacks=[st_cb])

        # Store & render the assistant's final answer
        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )
        st.write(response)

