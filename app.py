import os
import tempfile
import streamlit as st
from streamlit_chat import message
from local_gpt import LocalGPT

st.set_page_config(page_title="LocalGPT")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["process_input_spinner"] = st.empty()


def process_user_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_input = st.session_state["user_input"].strip()
        with st.session_state["process_input_spinner"], st.spinner("Processing..."):
            agent_response = st.session_state["assistant"].get_response(user_input)

        st.session_state["messages"].append((user_input, True))
        st.session_state["messages"].append((agent_response, False))
        st.session_state["user_input"] = ""


def process_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["process_file_spinner"], st.spinner(f"Processing... {file.name}"):
            st.session_state["assistant"].make_chain(file_path)
        os.remove(file_path)


def main_page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = LocalGPT()

    st.header("LocalGPT")

    st.subheader("Upload a PDF document.")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["process_file_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_user_input)


if __name__ == "__main__":
    main_page()
