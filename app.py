import streamlit as st
from langchain_groq import ChatGroq
from langchain_experimental.math import LLMMathChain
from langchain.chains import LLMChain           # still works for simple chains
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent



# Streamlit UI
st.set_page_config(page_title="Text To Math Solver & Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key.")
    st.stop()

# llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
llm = ChatOllama(model="gemma3")


# Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for facts and general knowledge.",
)

# Math solver tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve mathematical expressions.",
)

# Reasoning tool (LLMChain)
prompt = """
You are an agent who solves mathematical problems logically and provides clear step-by-step explanations.
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Answer reasoning-based questions and explain step-by-step."
)


# NEW: Create agent using LangGraph (replacement for initialize_agent)
assistant_agent = create_react_agent(
    llm=llm,
    tools=[wikipedia_tool, calculator_tool, reasoning_tool]
)


# Chat history setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a math assistant! Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# User input
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes..."
)

if st.button("Find my answer"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # NEW agent API — invoke with {"input": ...}
            response = assistant_agent.invoke(
                {"input": question},
                callbacks=[st_cb]
            )

            answer = response["output"]

            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.write("### Response:")
            st.success(answer)

    else:
        st.warning("Please enter a question.")

