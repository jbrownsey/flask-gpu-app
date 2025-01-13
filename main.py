# from flask import Flask, render_template, request
from quart import Quart, render_template, request
from hypercorn.config import Config
from hypercorn.asyncio import serve
from werkzeug.utils import secure_filename

from byaldi import RAGMultiModalModel
from io import BytesIO
import backoff
import asyncio
import json
import base64
import torch
import requests
from pyairtable import Api
from bubble_api import BubbleClient
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import pypdfium2 as pdfium
import urllib.request
import pymupdf
import os
import shapely
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from PIL import Image

from openai import OpenAIError
from openai import AsyncOpenAI, OpenAI

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode


from IPython.display import Image, display
from pydantic import BaseModel, Field

import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

import math

import uuid

import pprint

import getpass
import os
import threading

from pyngrok import ngrok, conf

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = getpass.getpass()

print("Enter your openAI API key: ")
api_key = getpass.getpass()

class evaluate_metric:
    #getting colpali capability
    RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1")
    #setting up API for gpt-4o
    MODEL = "gpt-4o"
    baseurl = "https://api.openai.com/v1"

    clienta = AsyncOpenAI(api_key=api_key,  base_url=baseurl)
    os.environ["OPENAI_API_BASE"] = baseurl
    os.environ["OPENAI_API_KEY"] = api_key
    @backoff.on_exception(backoff.expo, OpenAIError)
    def __init__(self):
        #needs to increase as webpage is used
        doc_id = 1

    def extract_pdf(self, pdf):
        keywords = ["water withdraw"]
        doc = pymupdf.open(pdf)
        pages=[]
        for i in range(doc.page_count):
            page = doc[i]
            for keyword in keywords:
                matches = page.search_for(keyword)
                if matches!=[]:
                    pages.append(i)
        if pages!=[]:
            pages = list(set(pages))
            pdf_new = pdfium.PdfDocument.new()
            pdf_old = pdfium.PdfDocument(pdf)
            pdf_new.import_pages(pdf_old,pages)
            extracted_filename = pdf[:-4]+' extracted.pdf'
            pdf_new.save(extracted_filename)
            return [True,extracted_filename]
        else:
            return [False]

    def get_page_as_png(self,extracted_file,year,j):
        evaluate_metric.RAG.index(input_path=extracted_file,
                index_name="multimodal_rag",
                store_collection_with_index=False,
                overwrite=True,)
        text_query = "What is the total water withdrawn by the company in "+year+"?"
        results = evaluate_metric.RAG.search(text_query,k=3)
        pages = convert_from_path(extracted_file)
        png_file = extracted_file[:-4]+' page.png'
        pages[results[j-1]['page_num']-1].save(png_file, 'PNG')
        return [pages[results[j-1]['page_num']-1],png_file]

    async def parse_page_with_gpt(self,base64_image: str) -> str:
        messages=[
            {
                "role": "system",
                "content": """
                
                You are a helpful assistant that extracts information from images.
                
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract information from image into text"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                        },
                    },
                ],
            }
        ]
        response = await evaluate_metric.clienta.chat.completions.create(
            model=evaluate_metric.MODEL,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def get_retriever(self,extracted_text):
        output_file_path = f"{uuid.uuid4()}.txt"
        
        with open(output_file_path, 'w') as json_file:
            json.dump(extracted_text, json_file, indent=2)
        
        print(f"data has been written to {output_file_path}")
        
        loader = TextLoader(output_file_path)
        data = loader.load()
        
        # Check if the file exists
        if os.path.exists(output_file_path):
            
            # Delete the file
            os.remove(output_file_path)
            print(f"File {output_file_path} deleted successfully.")
        else:
            print("File does not exist.")
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=500
        )
        doc_splits = text_splitter.split_documents(data)
        
        embedding_model = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(doc_splits, embedding_model) 
        retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        retriever_tool = create_retriever_tool(
            retriever,
            "document_understanding",
            "Retrieve and provide insights on document content analysis and knowledge extraction",
        )
        # tools = [retriever_tool]
        return retriever_tool

    #where does this need to be?
    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]

    ### Edges
    def grade_documents(self,state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        # Data model - where does this need to be?
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(grade)
        

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) then grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        print("question: ", question)
        print("context: ", docs)
        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "rewrite"
    
    ### Nodes
    def agent(self,state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
        model = model.bind_tools(tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    def rewrite(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}


    def generate(self,state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        print("context: ", docs)
        print("question: ", question)
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    def define_graph(self,retriever_tool):

        class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
            messages: Annotated[Sequence[BaseMessage], add_messages]
        
        # Define a new graph
        workflow = StateGraph(AgentState)
        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # Re-writing the question
        workflow.add_node("generate", self.generate)  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not
        workflow.add_edge(START, "agent")
        
        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        # Compile
        graph = workflow.compile()
        return graph

    def get_metric_value(self,graph,year):
        inputs = {
            "messages": [
                ("user", f"""
                        "Report the total water withdrawn by the company in {year}, if it is disclosed in the provided context. 
                        In your answer, print only the number (in metres cubed), and do not include units."""
                ),
            ]
        }
        return graph.stream(inputs)

    async def RAG_page(self,page_png,rep_period):
        buffered = BytesIO()
        page_png.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        extracted_text = await self.parse_page_with_gpt(img_base64)
        retriever_tool = self.get_retriever(extracted_text)
        global tools
        tools = [retriever_tool]
        graph = self.define_graph(retriever_tool)
        outputs = self.get_metric_value(graph,rep_period)
        for output in outputs:
            for key,value in output.items():
                if key=='generate':
                    answer = value
                else:
                    pass
        return answer

    async def obtain_and_upload(self,extracted_filename,rep_period,j,company_name):
        doc = pymupdf.open(extracted_filename)
        [page_png,png_filename] = self.get_page_as_png(extracted_filename,rep_period,j)
        answer = await self.RAG_page(page_png,rep_period)
        # upload_result = cloudinary.uploader.upload(png_filename,public_id=company_id +' '+rep_period)
        # update_report_table(report_id,upload_result['secure_url'],company_name +' '+rep_period +'.png',float(answer['messages'][0]))
        print("Metric was found on colpali page "+str(j))
        return str(float(answer['messages'][0]))

    #need to remove directories as well
    async def get_metric(self,pdf,company_name,rep_period):
        """Get metric from input pdf"""
        metric_appears = self.extract_pdf(pdf)
        if metric_appears[0]==True:
            extracted_filename = metric_appears[1]
            await self.obtain_and_upload(extracted_filename,rep_period,1,company_name)
        else:
            return "metric is not reported in document"

# Start flask app and set to ngrok
app = Quart(__name__)
# run_with_ngrok(app)
# port = "5000"

#remove ./?
UPLOAD_FOLDER = "/uploads"
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluator = evaluate_metric()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# Open a ngrok tunnel to the HTTP server
# public_url = ngrok.connect(port).public_url
# print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# # Update any base URLs to use the public ngrok URL
# app.config["BASE_URL"] = public_url

# ... Update inbound traffic via APIs to use the public-facing ngrok URL

# Function to start the Hypercorn server
# def start_hypercorn():
#     config = Config()
#     config.bind = ["127.0.0.1:8000"]  # Address and port to bind the server
#     asyncio.run(serve(app, config))  # Use asyncio to start Hypercorn

@app.route('/')
async def initial():
    return render_template("file1.html")

@app.route("/upload", methods=['POST'])
async def upload():
    if 'file-upload' not in request.files:
        return "No file part", 400
    
    file = request.files['file-upload']

    if file.filename=='':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        company_name = request.form["company_name"]
        reporting_period = request.form["reporting_year"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(filepath)
        # html_data = await evaluator.get_metric(filepath,company_name,reporting_period)
        html_data = await evaluator.get_metric(filepath,company_name,reporting_period)
        # return "Awaiting metric...", 202
        return render_template("file2.html", html_data=html_data)
    
    return "Invalid file type", 400


# if __name__ == '__main__':
#     app.run()

# Start the Flask server in a new thread
# threading.Thread(target=start_hypercorn).start()
if __name__ == "__main__":
    # Set up ngrok to forward to Hypercorn's port
    public_url = ngrok.connect(8000).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{8000}\"")
    # Update any base URLs to use the public ngrok URL
    app.config["BASE_URL"] = public_url
    config = Config()
    config.bind = ["127.0.0.1:8000"]  # Localhost and port for Hypercorn
    # Run Hypercorn in the main thread
    asyncio.run(serve(app, config))