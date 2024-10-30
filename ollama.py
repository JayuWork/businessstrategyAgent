import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="🤖 Ollama Chat Bot", page_icon="🤖")

st.header("🤖 Ollama :grey[Chat Bot]")

with st.chat_message("assistant"):
    st.write("Hello, how can I assist you today?")

prompt = st.chat_input("Enter your message here...")

llm = ChatOllama(model="llama3.1",temperature=0.7)

chain = llm | StrOutputParser()

if prompt:
    st.chat_message("user").write(prompt)
    response = chain.stream([HumanMessage(content=prompt)])
    with st.chat_message("assistant"):
        st.write_stream(response)
