import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import ollama

st.set_page_config(page_title="🤖 Ollama Chat Bot", page_icon="🤖")


"""
This is a simple chatbot that uses Ollama to generate responses to user messages.
"""
AVAILABLE_MODELS = ["llama3.1"]


def get_models():
    models = AVAILABLE_MODELS
    if not models:
        st.error("No models found. Please start Ollama server.")
        st.stop()
    models_list = []
    for model in models:
        models_list.append(model)

    return models_list


def init_session():

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            AIMessage(content="Hello, how can I assist you today?")
        ]

    for msg in st.session_state["chat_history"]:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)


def run_app():

    st.header("🤖 Ollama :grey[Chat Bot]")
    st.selectbox("Select a model", get_models(), key="selected_model")

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = get_models()[0]

    init_session()
    prompt = st.chat_input("Enter your message here...")

    selected_model = st.session_state["selected_model"]

    llm = ChatOllama(model=selected_model, temperature=0.7)

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
        response = llm.stream([HumanMessage(content=prompt)])

        with st.chat_message("assistant"):
            op = st.write_stream(response)

        st.session_state["chat_history"].append(AIMessage(content=op))


if __name__ == "__main__":
    run_app()
