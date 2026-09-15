"""ToolAgent — a Streamlit chat UI for an AI agent that can calculate,
look up Wikipedia, check the weather, and tell the date/time.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agent import run_agent, SYSTEM_PROMPT

load_dotenv()

st.set_page_config(page_title="ToolAgent", page_icon="🤖", layout="wide")
st.title("🤖 ToolAgent")
st.caption("An AI agent that decides when to calculate, search Wikipedia, check the weather, or check the date.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file, then restart the app.")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("trace"):
            with st.expander("🔧 Tools used"):
                for step in msg["trace"]:
                    st.markdown(f"**{step['tool']}**`({step['args']})`")
                    st.json(step["result"])

question = st.chat_input("Try: 'What's 18% of 2450?' or 'Weather in Pune?' or 'Who was Ada Lovelace?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, trace = run_agent(client, st.session_state.messages)
            st.write(answer)
            if trace:
                with st.expander("🔧 Tools used"):
                    for step in trace:
                        st.markdown(f"**{step['tool']}**`({step['args']})`")
                        st.json(step["result"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.display_messages.append({"role": "assistant", "content": answer, "trace": trace})
