"""
2. Real-time Agent Chat with Memory Injection
	•	Implement chat where users can interact with an agent.-> terminal based chatot for now. 
	•	As the user chats, memory updates in real time inside Zep.
	•	Ensure knowledge graph is updated asynchronously with relevant memory chunks.-> look into documentation for this. 

"""
import pdfplumber 
from dotenv import load_dotenv
import os 
import asyncio
from datetime import datetime, timezone
#now we call pydantic AI to extract the key knowledge points from the text. via LLM usage. [GROQ API KEY PROVIDED]
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import List
#load the respective libraries for zep memory layer. 
from zep_cloud.client import Zep
from zep_cloud.client import AsyncZep
import uuid
from zep_cloud.types import Message
import openai


#load the environment variables first: 
load_dotenv(".env")
load_dotenv()# back up measure. 

API_KEY = os.environ.get('ZEP_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


#define pydantic schemas for getting key knowledge points 
class keyknowledgeGraphPoints(BaseModel):
    """
    A array of key knowledge points extracted from the text. 

    -> returns a array of strings ,  where each string is a summarized key point. 
    """     
    structured_key_points : List[str] = Field(..., description="A list of all key points extracted from the text.")


#text processor agent. 
async def text_processing_agent(text_input):
    """
    we run this agent to extract the key knowledge points from the text. 
    """
    SYSTEM_PROMPT = """
    You are a helpful assistant that extracts key knowledge points from a given text.
    Your task is to analyze the text and identify the most important points that the author wants to convey.
    You should return a list of key points, each as a separate string.
    """
    agent = Agent(model='openai:gpt-4o-mini', output_type=keyknowledgeGraphPoints, system_prompt=SYSTEM_PROMPT)
    result = await agent.run(text_input)

    return result 

def scrape_text_from_pdf(pdf_file):
    """
    Scrape all the text from a given pdf file and return as a single string.
    """
    text={}
    with pdfplumber.open(pdf_file) as pdf:
        for i,page in enumerate(pdf.pages):
            text[i] = page.extract_text()
    return text 


def get_next_user_id(file_path="user_id.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            next_id = int(f.read().strip()) + 1
    else:
        next_id = 1
    with open(file_path, "w") as f:
        f.write(str(next_id))
    return f"user_{next_id}"


def user_intialization_process():
    #intialising the Zep connection object 
    client = Zep(api_key=API_KEY)

    user_id = get_next_user_id()
    print(f"Creating a new user with id {user_id}")

    #respectively intiate the new user in zep.
    new_user = client.user.add(
        user_id=user_id,
    )

    #create a new session id we get , 
    session_id = uuid.uuid4().hex

    #using the session id , we create  a session 
    client.memory.add_session(
        session_id=session_id,
        user_id=user_id,
    )

    return client,user_id,session_id



async def get_key_points_from_pdf(pdf_file):
    scraped_text = scrape_text_from_pdf(pdf_file)

    # Use the correct model name for OpenAI
    result = await text_processing_agent(scraped_text)
    return result.output.structured_key_points


#function to get group_id from the dict objet returned by scrape_text_from_pdf function. 
def get_unique_group_id(scraped_text):
    """
    get unique group id for each page of the pdf file. 
    """
    page_number = []
    page_values = [] # all scraped from the returend dict object. 
    # if the object is returned is a dict , extract the keys and return them as a list : 
    #then lets also have a list of values which are the text of the page.  
    for key,value in scraped_text.items():
        page_number.append(key)
        page_values.append(value)
    return page_number,page_values






#when building the knowledge graph . document id can be inferred from main and group id can be inferred from the get_unique_group_id function.
def add_to_knowledge_graph(user_id, client, page_numbers, page_texts, document_name):
    """
    Adds the raw text of each page to the knowledge graph, using group_id for each page.
    """
    for page_number, page_text in zip(page_numbers, page_texts):
        if not page_text:
            continue  # Skip empty pages
        group_id = f"{document_name}_{page_number}"
        new_episode = client.graph.add(
            group_id=group_id,
            #user_id=user_id,
            type="text",
            data=page_text
        )
        print(f"Added page {page_number} to graph as group {group_id}")
        print(f"Added episode: {new_episode}")
    print("Knowledge graph built successfully.")


async def chatbot_loop(client, user_id, session_id):
    print("\n--- Chatbot (type 'exit' to quit) ---\n")
    oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    user_name = user_id
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        # Add user message to Zep memory
        client.memory.add(
            session_id=session_id,
            messages=[Message(role_type="user", role=user_name, content=user_message)],
        )
        # Retrieve memory context
        memory = client.memory.get(session_id=session_id)
        system_message = """
You are a helpful assistant. Carefully review the facts about the user below and respond to the user's question. Be helpful and friendly.
"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": memory.context},
            {"role": "user", "content": user_message},
        ]
        response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        print("Assistant:", response.choices[0].message.content)


async def main():
    """
    main function to run the code. 
    """

    client,user_id,session_id = user_intialization_process()
    #now getting the scraped text, 
    pdf_file = "366_assg.pdf"
    #key_points = await get_key_points_from_pdf(pdf_file)
    scraped_text = scrape_text_from_pdf(pdf_file)
    page_number,page_values = get_unique_group_id(scraped_text)
    add_to_knowledge_graph(user_id, client, page_number, page_values, pdf_file)

    #now we run the chatbot. 
    await chatbot_loop(client, user_id, session_id)


#what if we could seperate different knowledge graphs for different pages of a pdf document ? 
#this would be a good way to go about it. 
#we can use the page number to seperate the knowledge graphs.  like a UNIQUE SEPERATION IDENTIFIER LOCALIZED FOR EACH PAGE. 
# A HASH MAP AI AGENT TO RETREIVE AND SUMMARIZE THE EMBEDDINGS WHEN NEEDED AND CALLED WITHIN THAT PAGE ?    