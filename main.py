# from flask import Flask, render_template, request
from quart import Quart, render_template, request, send_file
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
import shutil
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

print("Enter your airtable API key: ")
airtable_api_key = getpass.getpass()

api = Api(airtable_api_key)

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
            extracted_filename = pdf[:-4]+'_extracted.pdf'
            pdf_new.save(extracted_filename)
            return [True,extracted_filename]
        else:
            return [False]

    def get_page_as_png(self,extracted_file,year,j):
        text_query = "What is the total water withdrawn by the company in "+year+"?"
        index_dir = Path(extracted_file[:-4])
        index_config_file = index_dir / "index_config.json.gz"
        if index_config_file.exists():
            print(f"[INFO] Index already exists at {index_dir}. Loading existing index.")
            RAG = RAGMultiModalModel.from_index(index_dir)
        else:
            print(f"[INFO] No index found at {index_dir}. Creating a new index.")
            if index_dir.exists():
                print(f"[INFO] Removing stale index directory at {index_dir}.")
                shutil.rmtree(index_dir)
            evaluate_metric.RAG.index(input_path=extracted_file,
                index_name=extracted_file[:-4],
                store_collection_with_index=False,
            overwrite=True,)
            print(f"[INFO] New index created at {index_dir}. Loading it now.")
            RAG = RAGMultiModalModel.from_index(index_dir)
        print(f"[INFO] Using index at {index_dir} for search.")
        results = RAG.search(text_query,k=3)
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
    def grade_documents(self,state) -> Literal["generate", "rewrite", "end"]:
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
            state['irrelevant_count'] += 1
            if state['irrelevant_count'] >= 3:
                return "end"
            else:
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

    def end(self):
        print("---END---")

    def define_graph(self,retriever_tool):

        class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
            messages: Annotated[Sequence[BaseMessage], add_messages]
            irrelevant_count: int #track number of times deemed irrelevant
        # Define a new graph
        workflow = StateGraph(AgentState)
        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # Re-writing the question
        workflow.add_node("generate", self.generate)  # Generating a response after we know the documents are relevant
        workflow.add_node("end", self.end)  # End
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
        workflow.add_edge("end", END)
        
        # Compile
        graph = workflow.compile()
        return graph

    def get_metric_value(self,graph,year):
        inputs = {
            "messages": [
                # ("user", f"""
                #         "Report the total water withdrawn and the total water consumed by the company in {year}, in json format,
                #         if it is disclosed in the provided context. Give the numeric values in metres cubed and do not include units."""
                # ),
                ("user", f"""
                        Report the total water withdrawn by the company in {year}, if it is disclosed in the provided context. 
                        In your answer, print only the number (in metres cubed), and do not include units."""
                        ),
            ],
            "irrelevant_count":0
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
        try:
            answer = answer['messages'][0]
        except:
            answer = "skip"
        return [answer,img_base64]

    async def obtain_and_upload(self,extracted_filename,rep_period,j,company_name):
        doc = pymupdf.open(extracted_filename)
        page_png = self.get_page_as_png(extracted_filename,rep_period,j)
        [answer,img_base64] = await self.RAG_page(page_png[0],rep_period)
        # upload_result = cloudinary.uploader.upload(png_filename,public_id=company_id +' '+rep_period)
        # update_report_table(report_id,upload_result['secure_url'],company_name +' '+rep_period +'.png',float(answer['messages'][0]))
        if answer == "skip":
            print("Metric was not found")
        else:
            print("Metric was found on colpali page "+str(j))    
        #answer['messages'][0]
        return [answer,img_base64]
        #return [answer,img_base64]

    #need to remove directories as well
    async def get_metric(self,pdf,company_name,rep_period):
        """Get metric from input pdf"""
        metric_appears = self.extract_pdf(pdf)
        if metric_appears[0]==True:
            extracted_filename = metric_appears[1]
            return await self.obtain_and_upload(extracted_filename,rep_period,1,company_name)
        else:
            return "metric is not reported in document"

# Start flask app and set to ngrok
app = Quart(__name__)
# run_with_ngrok(app)
# port = "5000"

# connect to Reports Airtable

reports_table = api.table('appoixnEthALBAALd', 'tbl0JAnFkPP4vSY0R')

reports_dict = reports_table.all()

#convert to pandas dataframe
flat_dict = []
reports_id = []


for i in range(len(reports_dict)):

    reports_id.append(reports_dict[i]["id"])
    flat_dict.append(reports_dict[i]["fields"])

reports_df = pd.DataFrame.from_dict(flat_dict, orient = "columns")
reports_df.insert(0, "id", reports_id)

period = list(reports_df['Reporting Period'].values)
sus_reports = list(reports_df["Sustainability Report"].values)
comp_d = list(reports_df['Company'].values)
comp_names_d = list(reports_df['Name (autom. filled)'].values)
sector_d = list(reports_df['Sector (from Company)'].values)
ind = list(reports_df['Industry (from Company)'].values)
ids_d = list(reports_df['id'].values)

filenames = []
urls = []
companies = []
company_names = []
sectors = []
industries = []
rep_period = []
ids = []

for i in range(len(sus_reports)):
    try:
        x = sus_reports[i][0]["filename"]

        if ".pdf" in sus_reports[i][0]["filename"]:
            
            filenames.append(sus_reports[i][0]["filename"])
            urls.append(sus_reports[i][0]["url"])
            company_names.append(comp_names_d[i][0])
            companies.append(comp_d[i][0])
            ids.append(ids_d[i])
            sectors.append(sector_d[i][0])
            industries.append(ind[i][0])
            rep_period.append(period[i])

    except:
        pass

sus_reports_df = pd.DataFrame(zip(ids, companies, company_names, industries, sectors, rep_period, filenames, urls), columns = ["id", "Company", "Company name", "Industry", "Sector", "Reporting Period", "Filename", "URL"])

sectors_l = list(sus_reports_df['Sector'].values)

sector_industry_map = {}

for i in range(len(sectors_l)):
    sector_industry_map[sectors_l[i]] = list(set(sus_reports_df.loc[sus_reports_df['Sector'] == sectors_l[i]]['Industry'].values))

#remove ./?
UPLOAD_FOLDER = "/uploads/"

# Set the maximum allowed request size (in bytes)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 10 MB, adjust as needed

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluator = evaluate_metric()


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
    return await render_template("file3.html", sector_industry_map=json.dumps(sector_industry_map))

@app.route('/collected_pdfs/', methods=['POST'])
async def collected_pdfs():
    form = await request.form
    industry = form.get('industry')
    filtered_df = sus_reports_df.loc[sus_reports_df['Industry'] == industry]
    filtered_df = filtered_df[filtered_df["Reporting Period"]!="2021"].reset_index(drop = True)
    urls = list(filtered_df["URL"].values)
    ids = list(filtered_df["id"].values)
    rep_years = list(filtered_df["Reporting Period"].values)
    downloaded_files = []
    for i in range(len(urls)):
        try:
            url = urls[i]
            filename = str(ids[i])+'_'+str(rep_years[i])+'.pdf'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            urllib.request.urlretrieve(url,filepath)
            downloaded_files.append(filepath)
        except:
            pass
    # check if downloaded_files is empty
    if downloaded_files==[]:
        return "No files uploaded", 400
    return await render_template("file1.html", downloaded_files=downloaded_files)

@app.route("/show_result/", methods=['POST'])
async def show_result():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    images = []
    results = []
    companies = []
    reporting_periods = []
    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],file)
        #look up id to get company name and reporting period
        id_year = file.split('.')[0]
        print(id_year)
        id = id_year.split('_')[0]
        company_name = sus_reports_df.loc[sus_reports_df['id'] == id]['Company name'].values[0]
        reporting_period = sus_reports_df.loc[sus_reports_df['id'] == id]['Reporting Period'].values[0]
        #[html_data,img_base64] = await evaluator.get_metric(filepath,company_name,reporting_period)
        result = await evaluator.get_metric(filepath,company_name,reporting_period)
        print(result)
        # images.append(img_base64)
        # results.append(html_data)
        # companies.append(company_name)
        # reporting_periods.append(reporting_period)
    return await render_template("file2.html", results=results,images=images,companies=companies,reporting_periods=reporting_periods)

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
    config.timeout = 600
    # Run Hypercorn in the main thread
    asyncio.run(serve(app, config))