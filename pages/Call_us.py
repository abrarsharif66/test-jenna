import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
import time
from streamlit_lottie import st_lottie
import requests
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List

from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

#from utils import load_vector_store,similarity_search
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_photography = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

def load_vector_store(persist_directory: str) -> Chroma:
    """Load an existing Chroma vector store."""
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def similarity_search(vector_store: Chroma, query: str, k: int = 2) -> List:
    """Perform similarity search on the vector store."""
    results = vector_store.similarity_search(query, k=2)
    return results 
persist_directory = "chroma_db"
vector_store_z = load_vector_store(persist_directory)

st.title("Bravo Technologies")
st.write("Connect With our smart agent")
st_lottie(lottie_photography, height=300, key="photography")
time.sleep(2)

st.write("You are now talking to our agent Jenna")
# endcall=st.button("End call")
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

#using chat completion
conversation_history=[

            {
"role":"system",
                "content":"""You are Jenna a business development associate for Bravo Technologies, a leading software company. Your role is to have natural conversations with potential customers who are calling in to learn more about Bravo's products and services. 

            Since this interaction stems from a phone call, you should:

            - Greet the caller politely 
            - Ask for the caller's name and use it throughout the conversation  
            - Speak in a friendly yet professional tone
            - Allow the caller to drive the conversation by asking questions, but also be prepared to highlight Bravo's key offerings
            - Gather information about the caller's needs and interests related to software solutions
            - Provide relevant details about Bravo's products/services that could address their requirements
            - Avoid jargon and explain technical concepts in plain language
            - Make it clear you are an AI assistant having a friendly conversation, not an actual Bravo employee
            - End the call by thanking the person for their time and interest, and invite follow-up
            - your name and role is already introduced no need to reintroduce
            - keep the conversation natural speak more as it is needed else try to keep responses shorter

            Your goal is to have an engaging dialogue, build rapport, and position Bravo as an innovative solution for the caller's software needs."""
},
]





import speech_recognition as sr
import os
from groq import Groq
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def transcribe_and_process_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("User said:", text)
            if not text:  # Handle case where the text is empty or None
                return "I couldn't quite catch what you said. Can you please repeat?"
            return text

        # Handle case where speech is not recognized
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            return "I couldn't understand what you said. Could you try again?"

        # Handle issues related to API request errors
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
            return "There was an issue with the speech recognition service. Please try again later."

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_llm_response(user_input, history):
    
    history.append({"role": "user", "content": f"now using this {context} answer this user's query {user_input}"})

    # Call the LLM with the conversation history
    chat_completion = client.chat.completions.create(
        
        model="llama-3.1-70b-versatile",
        messages=history,
        temperature=0.3,
        max_tokens=100,
        stop=None,
        stream=False,
    )

    # Extract the LLM response and update the conversation history
    llm_response = chat_completion.choices[0].message.content
    history.append({"role": "assistant", "content": llm_response})

    return llm_response

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


def classify_convo(whole_chat):
    analyse_chat=[

                    {
        "role": "system",
        "content": """You will receive a chat history between a user and an LLM assistant in the following format:

        {"role":"system","content":"#instructions"}
        {"role":"user","content":"#message" },
        {"role":"assistant","content":"#response"}
        ...

        Your task is to analyze the overall sentiment of the chat history and strictly output one of the following:

        positive
        negative
        neutral

        Do not provide any explanations, sentences, or additional output beyond one of those three words."""
        },
        ]
    analyse_chat.append({"role": "user", "content": f"now classify this chat: {whole_chat}"})
    chat_completion = client.chat.completions.create(
        messages=analyse_chat,
        model="Llama3-70b-8192",
        temperature=0.1,
        max_tokens=100,
        top_p=1,
        stop=None,
        stream=False,
    )
    sentiment = chat_completion.choices[0].message.content
    return sentiment
def context_convo(whole_chat2):
    analyse_chat2=[

                    {
        "role": "system",
        "content": """From the conversation history between the user and the LLM assistant, analyze the topic or domain that the user appears to be most interested in or focused on. Your output should strictly be one of the following class names or sub-class names:

                class-cloudsolution
                - Applicationdevelopment
                - Devops
                - cloudcomputing
                class-ERP
                - CRM
                - SAPAMS
                class-Businessintelligence
                - AIandML
                class-Dataanalytics
                - Bigdata
                - Databasemigration
                - Powerplatform

                Do not provide any explanations or additional output beyond one of those class names or sub-class names representing the detected domain of interest."""
        },
        ]
    analyse_chat2.append({"role": "user", "content": f"now classify this chat: {whole_chat2}"})
    chat_completion = client.chat.completions.create(
        messages=analyse_chat2,
        model="Llama3-70b-8192",
        temperature=0,
        max_tokens=100,
        top_p=1,
        stop=None,
        stream=False,
    )
    domain = chat_completion.choices[0].message.content
    return domain   

import csv
from datetime import datetime

def write_to_csv(sentiment, domain):
    # Get the current month and year
    current_month = datetime.now().strftime("%B_%Y")
    
    # Create the filename with the current month and year
    filename = f"{current_month}.csv"
    
    # Check if the file exists, if not, create it with headers
    try:
        with open(filename, 'x', newline='') as csvfile:
            fieldnames = ['Sentiment', 'Domain']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass
    
    # Open the file in append mode and write the sentiment and domain
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentiment, domain])

# Conversational loop
fi=0

while True:
    if fi==0:
        text_to_speech("Hi there! I'm jenna Business develeopment associate from bravo technologies, How may i help you today?")
        fi+=1
    try:
     user_input = transcribe_and_process_input()
    except:
        text_to_speech("I'm sorry i couldn't catch what you said can you repeat?")
        user_input = transcribe_and_process_input()
    queryz = user_input
    context=similarity_search(vector_store_z,queryz)
    print("printing context`")
    print(context)
    if "goodbye" in user_input.lower():
        print("Goodbye!")
        by="goodbye have a great day"
        text_to_speech(by)
        print("Classifying...")
        sentiment=classify_convo(conversation_history)
        print(f"the overall conversation is {sentiment}")
        print("analysing..")
        conv_domain=context_convo(conversation_history)
        print(f"the class is = {conv_domain}")
        print("writing to csv")
        write_to_csv(sentiment,conv_domain)
        break
    
    print("transcribe done")
    llm_response = get_llm_response(user_input, conversation_history)
    print("llm response received")
    text_to_speech(llm_response)



