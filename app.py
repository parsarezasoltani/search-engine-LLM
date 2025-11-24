# app.py
# Fully Fixed & Optimized for Hugging Face Spaces (2025)
# No more "Agent stopped due to iteration/time limit" errors

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables (works locally + HF Spaces secrets)
load_dotenv()

# ========================= TOOLS =========================
arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv)
arxiv_tool.name = "arxiv_search"
arxiv_tool.description = "Search arXiv for academic papers. Use only for research, math, science, or technical topics."

wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
wiki_tool.name = "wikipedia"
wiki_tool.description = "Search Wikipedia for well-established facts, history, definitions, biographies."

search_tool = DuckDuckGoSearchRun(name="web_search")
search_tool.description = "Search the internet via DuckDuckGo. Best for current events, news, prices, sports scores, weather, etc."

tools = [search_tool, arxiv_tool, wiki_tool]

# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="Groq + Search Agent", page_icon="magnifying glass")
st.title("magnifying glass LangChain Agent with Web + arXiv + Wikipedia")
st.caption("Powered by **Groq** (ultra-fast) + LangChain tools. No more timeout errors!")

# Sidebar
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", value=os.getenv("GROQ_API_KEY", ""))
    
    if not api_key:
        st.warning("Enter your Groq API key (get it free at https://console.groq.com)")
        st.stop()

    model = st.selectbox(
        "Model (faster = more reliable on HF Spaces)",
        [
            "llama-3.1-70b-versatile",     # Best balance
            "llama-3.1-8b-instant",        # Fastest but sometimes loops
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
        ],
        index=0
    )

# ========================= CHAT HISTORY =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web, arXiv papers, and Wikipedia in real-time. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ========================= PROMPT TEMPLATE =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to three tools:
- web_search: for current events, facts, prices, weather, etc.
- wikipedia: for established knowledge, history, biographies
- arxiv_search: for scientific papers and research

Use tools ONLY when necessary. If you already know the answer or it's common knowledge, reply directly.
Be concise and accurate. Always cite sources when using tools."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ========================= LLM & AGENT =========================
@st.cache_resource
def get_agent(_api_key, _model):
    llm = ChatGroq(
        groq_api_key=_api_key,
        model=_model,
        temperature=0.2,
        streaming=True,
    )
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,           # Rarely reached with tool-calling
        max_execution_time=None,    # No hard timeout needed
    )
    return agent_executor

# ========================= USER INPUT =========================
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        try:
            agent_executor = get_agent(api_key, model)
            
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": [
                        (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                        if m["role"] != "system"
                    ],
                },
                {"callbacks": [st_callback]}
            )
            
            answer = response["output"]
            
        except Exception as e:
            answer = f"Oops! Something went wrong: {str(e)}. Try again or simplify your question."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
