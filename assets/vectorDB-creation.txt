import logging
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyMuPDFLoader, 
    UnstructuredMarkdownLoader
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from typing import TypedDict

from dotenv import (
    load_dotenv, 
    find_dotenv
)
from pydantic import BaseModel, Field
import os
import warnings
warnings.filterwarnings("ignore")


load_dotenv(
    find_dotenv()
)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def createVectorStoreAndLLMFunctionForPDFFiles(data_path, **kwargs) :

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    logger.info("[INFO] ----> Loading the pipeline configuration - LLM and Embeddings Model.")
    
    if kwargs["llm_model_provider"] == "Google" :
        logger.info("[INFO] ----> Pipeline LLM Model :- Google Gemini LLM.")
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-001", 
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            max_output_tokens=kwargs["maximum_output_tokens"],
        )
    elif kwargs["llm_model_provider"] == "OpenAI" :
        logger.info("[INFO] ----> Pipeline LLM Model :- OpenAI GPT-4 LLM.")
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=kwargs["temperature"], 
            top_p=kwargs["top_p"],
            max_completion_tokens=kwargs["maximum_output_tokens"]
        )
    else :
        logger.info("[INFO] ----> Pipeline LLM Model :- Default --> OpenAI GPT-4 LLM.")
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=kwargs["temperature"], 
            top_p=kwargs["top_p"],
            max_completion_tokens=kwargs["maximum_output_tokens"]
        )
    if kwargs["embeddings_provider"] == "Google" :
        logger.info("[INFO] ----> Pipeline Embeddings :- Google Gemini Embeddings.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    elif kwargs["embeddings_provider"] == "Huggingface" :

        logger.info("[INFO] ----> Pipeline Embeddings :- HuggingFace Embeddings.")
        # Define the path to the pre-trained model you want to use
        modelPath = "sentence-transformers/all-MiniLM-L12-v2"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        model_kwargs = {'device':'cpu'}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )

    logger.info("[INFO] ----> Models loaded successfully.")

    CHROMA_DB_DIR = kwargs["chroma_db_persist_directory"]


    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR) :
        logger.info("[INFO] ----> VectorDB Exists already no need for creating again....\n")
        vectorDB = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
        )
    else :
        logger.info("[INFO] ----> Creating the vectorstore for the dir:- {}\n".format(CHROMA_DB_DIR))
        documents = DirectoryLoader(
            path="{}\\".format(data_path), 
            glob="*.pdf", 
            loader_cls = PyMuPDFLoader
        ).load()

        logger.info("[INFO] ----> Total documents loaded are :- {} Documents.\n".format(len(documents)))

        splitted_documents = RecursiveCharacterTextSplitter(
            chunk_size=kwargs["chunk_length"], 
            chunk_overlap=kwargs["chunk_overlap_length"], 
            length_function=len,
        ).split_documents(documents)

        logger.info("[INFO] ----> Document chunks created are :- {} chunks.\n".format(len(splitted_documents)))

        logger.info("[INFO] ----> Creating the vectorstore, please wait.....")
        vectorstore = Chroma.from_documents(
            documents, embeddings, persist_directory=CHROMA_DB_DIR
        )
        vectorDB = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
        )
        logger.info("[INFO] ----> Vectorstore created successfully....")

    return llm, embeddings, vectorDB, logger



def createVectorStoreAndLLMFunctionForMarkdownFiles(data_path, **kwargs) :

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    logger.info("[INFO] ----> Loading the pipeline configuration - LLM and Embeddings Model.\n")
    
    # Initialize LLM
    if kwargs["llm_model_provider"] == "Google" :

        logger.info("[INFO] ----> Pipeline LLM Model :- Google Gemini LLM.")
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-001", 
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            max_output_tokens=kwargs["maximum_output_tokens"],
        )

    elif kwargs["llm_model_provider"] == "OpenAI" :

        logger.info("[INFO] ----> Pipeline LLM Model :- OpenAI GPT-4 LLM.")
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=kwargs["temperature"], 
            top_p=kwargs["top_p"],
            max_completion_tokens=kwargs["maximum_output_tokens"]
        )

    else :

        logger.info("[INFO] ----> Pipeline LLM Model :- Default --> OpenAI GPT-4 LLM.")
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=kwargs["temperature"], 
            top_p=kwargs["top_p"],
            max_completion_tokens=kwargs["maximum_output_tokens"]
        )

    if kwargs["embeddings_provider"] == "Google" :
        logger.info("[INFO] ----> Pipeline Embeddings :- Google Gemini Embeddings.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    elif kwargs["embeddings_provider"] == "Huggingface" :
        logger.info("[INFO] ----> Pipeline Embeddings :- HuggingFace Embeddings.")
        # Define the path to the pre-trained model you want to use
        modelPath = "sentence-transformers/all-MiniLM-L12-v2"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        model_kwargs = {'device':'cpu'}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )

    logger.info("[INFO] ----> Models loaded successfully.")

    CHROMA_DB_DIR = kwargs["chroma_db_persist_directory"]


    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR) :
        logger.info("[INFO] ----> VectorDB Exists already no need for creating again....\n")
        vectorDB = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
        )
    else :
        logger.info("[INFO] ----> Creating the vectorstore for the dir:- {}\n".format(CHROMA_DB_DIR))
        markdown_path = data_path
        print("[INFO] ----> Document Loading ongoing, please wait.....")
        loader = DirectoryLoader(
            markdown_path, 
            glob='./*.md', 
            loader_cls=UnstructuredMarkdownLoader
        )
        documents = loader.load()
        print("[INFO] ----> Document Loading completed with count :- {}\n".format(len(documents)))

        logger.info("[INFO] ----> Total documents loaded are :- {} Documents.\n".format(len(documents)))

        splitted_documents = RecursiveCharacterTextSplitter(
            chunk_size=kwargs["chunk_length"], 
            chunk_overlap=kwargs["chunk_overlap_length"], 
            length_function=len,
        ).split_documents(documents)

        logger.info("[INFO] ----> Document chunks created are :- {} chunks.\n".format(len(splitted_documents)))

        logger.info("[INFO] ----> Creating the vectorstore, please wait.....")
        vectorstore = Chroma.from_documents(
            documents, embeddings, persist_directory=CHROMA_DB_DIR
        )
        vectorDB = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
        )
        logger.info("[INFO] ----> Vectorstore created successfully....")

    return llm, embeddings, vectorDB, logger

