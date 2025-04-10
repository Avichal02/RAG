import streamlit as st

def generate_bot_reply(user_input):
    # Your chatbot logic here
    return "You said: " + user_input

st.title("🤖 Chatbot")

user_input = st.text_input("You:")

if user_input:
    response = generate_bot_reply(user_input)
    st.text_area("Bot:", value=response, height=100)
