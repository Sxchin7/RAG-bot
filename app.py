import streamlit as st
import tempfile
from rag_engine import load_and_process_pdf,vectorize,get_answer,find_similarity
st.set_page_config(page_title="Chat", page_icon="⚡")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
file=st.file_uploader("Upload your PDF file")
if file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix=f".{file.name.split('.')[-1]}") as temp_file:
        temp_file.write(file.getvalue())
        temp_path=temp_file.name

    if "vector_storage" not in st.session_state:
            with st.spinner("Processing PDF..."):
                chunks = load_and_process_pdf(temp_path)
                st.session_state["vector_storage"] = vectorize(chunks)
            st.success("Ready! Ask me anything.")
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])
if "vector_storage" in st.session_state:
    prompt=st.chat_input(placeholder="Your message")
    if prompt:
            # bot = st.chat_message("Bot", avatar="ai", width="stretch")
            user = st.chat_message("User", avatar="user")
            user.write(prompt)
            similar=find_similarity(prompt,st.session_state["vector_storage"])
            answer=get_answer(prompt,similar)
            bot = st.chat_message("Bot", avatar="ai")
            bot.write(answer)
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.session_state["messages"].append({"role": "assistant", "content": answer})