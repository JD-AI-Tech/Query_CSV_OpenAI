import streamlit as st
import os
import pandas as pd
import logging

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from file_utility import create_directory

os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]
number_of_records_to_display = 151
data_directory = 'data'
create_directory(data_directory)
log_directory = 'log'
create_directory(log_directory)

# setup UI
st.header("Query CSV file using natural language powered by OpenAI GPT-3.5 or Davinci")
genre = st.radio(
    "Select the [language model](https://getgenie.ai/davinci-vs-turbo/)  that you want to use.",
    ('gpt-3.5-turbo','text-davinci-003'))

model_name = 'gpt-3.5-turbo'
if genre == 'gpt-3.5-turbo':
    model_name = 'gpt-3.5-turbo'
else:
    model_name = 'text-davinci-003'

with st.sidebar:
    st.title('About')
    st.markdown('''
        This is a Proof Of Concept (POC). 
        The goal is to query a CSV file using natural language.
        - OpanAI's GPT API queries CVS file
        - LangChain connects to CVS file
 
     ''')
    st.title('Technology')
    st.markdown('''
        Developed by Jorge Duenas using:
        - [OpenAI GPT-3.5 API](https://openai.com/product)
        - [Streamlit.io](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/en/latest/index.html)
        - [Python](https://www.python.org/)
        - [Anaconda](https://www.anaconda.com/)   
        - [Pycharm IDE](https://www.jetbrains.com/pycharm/) 
        - [Kaggle CSV datasets](https://www.kaggle.com/datasets?fileType=csv)    
    ''')
# UI
uploaded_file = st.file_uploader('', type=['csv'])
if uploaded_file is not None:
    with open(os.path.join(data_directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    saved_file_name = data_directory + "/" + uploaded_file.name

    # read in file contents
    df = pd.read_csv(saved_file_name)

    user_input = st.text_input("Query CSV file using natural language: (example: How  many row are in the csv file?)")
    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['user_input'],
        template='display all the colum data along with  {user_input}'
    )

    csv_expander = st.expander("Peek at the top " + str(number_of_records_to_display - 1) + " rows of the CSV file")
    csv_expander.write(df.head(number_of_records_to_display))

    # Create a question-answering chain using # text-davinci-003 #gpt-3.5-turbo
    llm = OpenAI(temperature=0, model_name=model_name)
    memory = ConversationBufferMemory(input_key=user_input, memory_key='chat_history')
    agent = create_csv_agent(llm,
                             saved_file_name,
                             verbose=True,
                             memory=memory)

    if user_input:
        response = agent.run(user_input)
        st.write(response)

        # with st.expander('History'):
        #     st.info(memory.buffer)
