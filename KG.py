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
from memory_tool import MemoryTool, MemoryQuery, Memory
import json
from health_data_tool import HealthData, HealthDataTool


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

    # Always generate a unique user_id and session_id
    user_id = f"user_{uuid.uuid4().hex}"
    print(f"Creating a new user with id {user_id}")

    #respectively intiate the new user in zep.
    client.user.add(
        user_id=user_id,
    )

    #create a new session id we get , 
    session_id = uuid.uuid4().hex

    #using the session id , we create  a session 
    client.memory.add_session(
        session_id=session_id,
        user_id=user_id,
    )

    client.memory.add(
        session_id=session_id,
        messages=[Message(role_type="system", role="system", content="The user uploaded a PDF: 366_assg.pdf")]
    )

    client.memory.add(
        session_id=session_id,
        messages=[Message(role_type="system", role="system", content="The user's health data for 2024-06-01 has been added.")]
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
def add_to_knowledge_graph(client, page_texts, document_name):
    """
    Adds the raw text of each page to the knowledge graph, using group_id for each page.
    """
    combined_texts = [page_text for page_text in page_texts if page_text]
    if not combined_texts:
        print("No non-empty pages to add to the knowledge graph.")
        return
    combined_text = "\n\n".join(combined_texts)
    group_id = f"{document_name}_all_pages"
    new_episode = client.graph.add(
        group_id=group_id,
        type="text",
        data=combined_text
    )
    print(f"Added all pages to graph as group {group_id}")
    print(f"Added episode: {new_episode}")
    print("Knowledge graph built successfully.")


def add_health_data_to_kg(client, health_data):
    group_id = f"health_{health_data['user_id']}_{health_data['date']}"
    data = json.dumps(health_data)
    result = client.graph.add(
        group_id=group_id,
        type="json",
        data=data
    )
    print(f"Added health data to graph as group {group_id}")
    print(f"Added episode: {result}")


async def chatbot_loop(client, user_id, session_id, pdf_context, health_context):
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
        # Retrieve the latest memory context (this includes KG facts/entities)
        memory = client.memory.get(session_id=session_id)
        context = getattr(memory, "context", "")
        system_message = f"""
        You are a helpful assistant.\n\nHere is the content of the user's PDF:\n{pdf_context}\n\nHere is the user's health data:\n{health_context}\n\nHere is the user's session memory context:\n{context}\n\nCarefully review the facts about the user below and respond to the user's question. Be helpful and friendly.\n"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        print("Assistant:", response.choices[0].message.content)


async def use_memory_tool(client, user_id):
    # Initialize the tool
    memory_tool = MemoryTool(client)

    # When you need context
    query = MemoryQuery(query="user's question here")
    results = memory_tool.search_memories(query)

    # When you want to store new information
    new_memory = Memory(
        content="Important conversation point",
        metadata={"context": "user_interaction"}
    )
    memory_tool.add_memory(new_memory)

def get_pdf_kg_content(client, document_name):
    group_id = f"{document_name}_all_pages"
    search_results = client.graph.search(query="*", group_id=group_id, limit=1)
    results = getattr(search_results, 'results', [])
    if results:
        return getattr(results[0], 'text', '')
    return ""

def get_health_kg_content(client, user_id, date):
    group_id = f"health_{user_id}_{date}"
    search_results = client.graph.search(query="*", group_id=group_id, limit=1)
    results = getattr(search_results, 'results', [])
    if results:
        return getattr(results[0], 'text', '')  # or 'data' if the SDK returns the JSON as 'data'
    return ""


def main():
    client, user_id, session_id = user_intialization_process()
    pdf_file = "366_assg.pdf"
    scraped_text = scrape_text_from_pdf(pdf_file)
    page_values = list(scraped_text.values())
    add_to_knowledge_graph(client, page_values, pdf_file)
    with open("mock_health_data.json", "r") as f:
        health_json = json.load(f)
    add_health_data_to_kg(client, health_json)
    # Retrieve KG data for prompt injection
    pdf_context = get_pdf_kg_content(client, pdf_file)
    health_context = get_health_kg_content(client, health_json['user_id'], health_json['date'])
    # Start chatbot loop with injected context
    asyncio.run(chatbot_loop(client, user_id, session_id, pdf_context, health_context))




if __name__ == "__main__":
    main()