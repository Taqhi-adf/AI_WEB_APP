import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
#from langchain.schema import AIMessage, HumanMessage


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("AI Chatbot")
st.subheader("Built with Streamlit, LangChain, and GPT-4o")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for conversation chain
if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    # Ensure return_messages=True is set correctly
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm, memory=memory, verbose=False
    )

# Render existing chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input from user
user_input = st.chat_input("Type Your Message here...")

if user_input:
    # Display user message instantly
    with st.chat_message("user"):
        st.write(user_input)

    # Append to chat history state
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Generate response from the conversation chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Lowercase .conversation to match initialization
            response = st.session_state.conversation.predict(input=user_input)
            st.write(response)

    # Append AI response to chat history state
    st.session_state.chat_history.append(AIMessage(content=response))

# Sidebar options
with st.sidebar:
    st.title("Options")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        # Re-initialize the chain and memory on clear
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.conversation = ConversationChain(
            llm=llm, memory=memory, verbose=False
        )
        st.rerun()

    st.subheader("About")
    st.markdown(
        """
        This chatbot uses:
        - **Streamlit** for the web interface
        - **LangChain** for the conversation routing
        - **GPT-4o** as the language model
        - **Conversation Buffer Memory** to retain context
        """
    )