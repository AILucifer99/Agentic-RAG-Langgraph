from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import json
from collections import defaultdict, Counter

# Import your existing modules
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from typing import TypedDict
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for analytics and chat sessions
chat_sessions = {}
analytics_data = {
    'total_queries': 0,
    'dataops_queries': 0,
    'itops_queries': 0,
    'offtopic_queries': 0,
    'avg_response_time': 0,
    'query_history': [],
    'hourly_stats': defaultdict(int),
    'daily_stats': defaultdict(int),
    'route_confidence': [],
    'error_count': 0,
    'user_sessions': set()
}

# Define structured output model
class ClassificationOutput(BaseModel):
    category: str = Field(description="The classification label (dataops or itops)")
    confidence: float = Field(description="Confidence score between 0 and 1")

# Define the state structure
class MyGraphState(TypedDict):
    query: str
    route: str
    confidence: float
    response: str

class RAGPipeline:
    def __init__(self):
        self.llm = None
        self.vectorstore = None
        self.routing_chain = None
        self.dataops_prompt = None
        self.itops_prompt = None
        self.app_graph = None
        self.initialized = False
    
    def initialize(self, config):
        """Initialize the RAG pipeline with given configuration"""
        try:
            load_dotenv(find_dotenv())
            
            # Pipeline configuration
            if config['model_provider'] == "Google":
                os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-001",
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    max_output_tokens=config['max_tokens']
                )
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
            elif config['model_provider'] == "OpenAI":
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    max_completion_tokens=config['max_tokens']
                )
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Fixed paths for simplicity
            CHROMA_DB_DIR = "vector-store/AWS-Bedrock-API-Lambda-Glue-Opensearch-DB-Google-Embeddings"
            data_folder_path = "Data"
            
            if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
                self.vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            else:
                documents = DirectoryLoader(
                    path=f"{data_folder_path}/",
                    glob="*.pdf",
                    loader_cls=PyMuPDFLoader
                ).load()
                
                splitted_documents = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=256
                ).split_documents(documents)
                
                vectorstore = Chroma.from_documents(
                    documents,
                    embeddings,
                    persist_directory=CHROMA_DB_DIR
                )
                self.vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            
            # Setup routing
            routing_template = """
            You are an intelligent routing agent. Classify user queries into: 'dataops', 'itops', or 'off-topic'.
            Provide a confidence score between 0 and 1.
            If the query deals with AWS S3, AWS Glue, AWS Bedrock and AWS Opensearch
            then classify as 'dataops',
            if the query deals with AWS API Gateway and AWS Lambda services, then classify it as 'itops'.

            Examples:
            Q: Explain the concept of AWS S3 bucket creation? → dataops (confidence: 0.95)
            Q: Pricing about the AWS Lambda services and how to develop applications using it. → itops (confidence: 0.98)
            Q: What's the weather today? → off-topic (confidence: 0.99)

            The confidence score should be based on:
            - Semantic similarity to the examples provided.
            - Clarity and specificity of the query.
            - Presence of domain-specific keywords (e.g., 's3', 'lambda', 'opensearch', 'api gateway', etc.).
            - Contextual alignment with typical tasks in data operations or IT operations.
            - If the query is unrelated to either domain, classify it as 'off-topic'.
           
            Now classify:
            Q: {query}
            """
            
            routing_prompt = PromptTemplate(input_variables=["query"], template=routing_template)
            structured_llm = self.llm.with_structured_output(ClassificationOutput)
            self.routing_chain = routing_prompt | structured_llm
            
            # System prompts
            self.dataops_prompt = """You are a veteran professional expert in Data Ops.
            Provide accurate response about AWS S3, AWS Glue, AWS Bedrock and AWS Opensearch services.
            So use the context provided below and answer the question accurately.
            Always try to give a complete answer within 100 words and use professional english tonality.
            """
            
            self.itops_prompt = """You are a veteran professional expert in IT Ops.
            Provide accurate response about AWS Lambda and AWS API Gateway.
            So use the context provided below and answer the question accurately.
            Always try to give a complete answer within 100 words and use professional english tonality."""
            
            # Build the graph
            self._build_graph()
            self.initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Pipeline initialization error: {str(e)}")
            return False
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        graph = StateGraph(MyGraphState)
        graph.add_node("intelligent_user_query_router", self._route_query)
        graph.add_node("dataops_agent", self._handle_dataops)
        graph.add_node("itops_agent", self._handle_itops)
        graph.add_node("offtopic_agent", self._handle_offtopic)
        
        graph.set_entry_point("intelligent_user_query_router")
        graph.add_conditional_edges(
            "intelligent_user_query_router",
            lambda state: state["route"],
            {
                "dataops": "dataops_agent",
                "itops": "itops_agent",
                "off-topic": "offtopic_agent"
            }
        )
        graph.add_edge("dataops_agent", END)
        graph.add_edge("itops_agent", END)
        graph.add_edge("offtopic_agent", END)
        
        self.app_graph = graph.compile()
    
    def _emit_step(self, session_id, step_name, details, status="running"):
        """Emit agent step to frontend"""
        step_data = {
            "step": step_name,
            "details": details,
            "status": status,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        socketio.emit('agent_step', step_data, room=session_id)
    
    def _search_vector_db(self, session_id, query, num_documents=5):
        """Search vector database"""
        try:
            self._emit_step(session_id, "🔍 Vector Search", f"Searching for {num_documents} relevant documents", "running")
            docs = self.vectorstore.similarity_search(query, k=num_documents)
            self._emit_step(session_id, "🔍 Vector Search", f"Found {len(docs)} relevant documents", "completed")
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            self._emit_step(session_id, "🔍 Vector Search", f"Error: {str(e)}", "error")
            return ""
    
    def _route_query(self, state):
        """Route query to appropriate agent"""
        try:
            session_id = state.get('session_id')
            self._emit_step(session_id, "🧭 Query Router", "Analyzing query intent and routing", "running")
            
            query = state["query"]
            result = self.routing_chain.invoke({"query": query})
            route = result.category.lower()
            confidence = result.confidence
            
            state["route"] = route
            state["confidence"] = confidence
            
            # Update analytics
            analytics_data['route_confidence'].append(confidence)
            
            self._emit_step(session_id, "🧭 Query Router",
                          f"Routed to: {route.upper()} (confidence: {confidence:.2f})",
                          "completed")
            return state
        except Exception as e:
            analytics_data['error_count'] += 1
            self._emit_step(state.get('session_id'), "🧭 Query Router", f"Error: {str(e)}", "error")
            return {"error": str(e)}
    
    def _handle_dataops(self, state):
        """Handle DataOps queries"""
        try:
            session_id = state.get('session_id')
            self._emit_step(session_id, "📊 DataOps Agent", "Processing DataOps query", "running")
            context = self._search_vector_db(session_id, state["query"])
            
            self._emit_step(session_id, "🤖 LLM Generation", "Generating DataOps response", "running")
            response = self.llm.invoke(
                f"{self.dataops_prompt}\nContext:\n{context}\n\nQuestion: {state['query']}"
            )
            state["response"] = response
            
            # Update analytics
            analytics_data['dataops_queries'] += 1
            
            self._emit_step(session_id, "📊 DataOps Agent", "DataOps response generated", "completed")
            return state
        except Exception as e:
            analytics_data['error_count'] += 1
            self._emit_step(state.get('session_id'), "📊 DataOps Agent", f"Error: {str(e)}", "error")
            return {"error": str(e)}
    
    def _handle_itops(self, state):
        """Handle ITOps queries"""
        try:
            session_id = state.get('session_id')
            self._emit_step(session_id, "🖥️ ITOps Agent", "Processing ITOps query", "running")
            context = self._search_vector_db(session_id, state["query"])
            
            self._emit_step(session_id, "🤖 LLM Generation", "Generating ITOps response", "running")
            response = self.llm.invoke(
                f"{self.itops_prompt}\nContext:\n{context}\n\nQuestion: {state['query']}"
            )
            state["response"] = response
            
            # Update analytics
            analytics_data['itops_queries'] += 1
            
            self._emit_step(session_id, "🖥️ ITOps Agent", "ITOps response generated", "completed")
            return state
        except Exception as e:
            analytics_data['error_count'] += 1
            self._emit_step(state.get('session_id'), "🖥️ ITOps Agent", f"Error: {str(e)}", "error")
            return {"error": str(e)}
    
    def _handle_offtopic(self, state):
        """Handle off-topic queries"""
        session_id = state.get('session_id')
        self._emit_step(session_id, "🚫 Off-topic Handler", "Query is off-topic", "completed")
        state["response"] = "I can only answer DataOps and ITOps related questions."
        
        # Update analytics
        analytics_data['offtopic_queries'] += 1
        
        return state
    
    def process_query(self, query, session_id):
        """Process a query through the pipeline"""
        if not self.initialized:
            return {"error": "Pipeline not initialized"}
        
        start_time = time.time()
        
        try:
            # Update analytics
            analytics_data['total_queries'] += 1
            current_hour = datetime.now().hour
            current_date = datetime.now().strftime('%Y-%m-%d')
            analytics_data['hourly_stats'][current_hour] += 1
            analytics_data['daily_stats'][current_date] += 1
            
            # Add session tracking
            state = {
                "query": query,
                "session_id": session_id
            }
            
            # Process through graph
            response_state = self.app_graph.invoke(state)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update analytics
            analytics_data['avg_response_time'] = (
                (analytics_data['avg_response_time'] * (analytics_data['total_queries'] - 1) + response_time) /
                analytics_data['total_queries']
            )
            
            # Store query history
            analytics_data['query_history'].append({
                'query': query,
                'route': response_state.get('route', 'unknown'),
                'confidence': response_state.get('confidence', 0),
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            })
            
            # Keep only last 1000 queries
            if len(analytics_data['query_history']) > 1000:
                analytics_data['query_history'] = analytics_data['query_history'][-1000:]
            
            return response_state
            
        except Exception as e:
            analytics_data['error_count'] += 1
            logging.error(f"Query processing error: {str(e)}")
            return {"error": str(e)}

# Initialize pipeline instance
rag_pipeline = RAGPipeline()

@app.route('/')
def index():
    """Main application page"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        analytics_data['user_sessions'].add(session['session_id'])
    
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    return render_template('analytics.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_pipeline():
    """Initialize the RAG pipeline"""
    try:
        config = request.json
        success = rag_pipeline.initialize(config)
        
        if success:
            return jsonify({"status": "success", "message": "Pipeline initialized successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to initialize pipeline"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a user query"""
    try:
        data = request.json
        query = data.get('query', '')
        session_id = session.get('session_id')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        if not rag_pipeline.initialized:
            return jsonify({"error": "Pipeline not initialized"}), 400
        
        # Process query
        result = rag_pipeline.process_query(query, session_id)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Extract response content
        response = result.get("response", "No response generated.")
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        return jsonify({
            "response": response_text,
            "route": result.get("route", "unknown"),
            "confidence": result.get("confidence", 0)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data"""
    try:
        # Calculate additional metrics
        total_queries = analytics_data['total_queries']
        
        # Route distribution
        route_distribution = {
            'dataops': analytics_data['dataops_queries'],
            'itops': analytics_data['itops_queries'],
            'offtopic': analytics_data['offtopic_queries']
        }
        
        # Recent queries (last 24 hours)
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        recent_queries = [
            q for q in analytics_data['query_history']
            if datetime.fromisoformat(q['timestamp']) > last_24h
        ]
        
        # Hourly distribution for chart
        hourly_data = []
        for hour in range(24):
            hourly_data.append({
                'hour': f"{hour:02d}:00",
                'queries': analytics_data['hourly_stats'].get(hour, 0)
            })
        
        # Daily data for the last 7 days
        daily_data = []
        for i in range(7):
            date = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            daily_data.append({
                'date': date,
                'queries': analytics_data['daily_stats'].get(date, 0)
            })
        daily_data.reverse()
        
        # Confidence scores
        avg_confidence = sum(analytics_data['route_confidence']) / len(analytics_data['route_confidence']) if analytics_data['route_confidence'] else 0
        
        return jsonify({
            'total_queries': total_queries,
            'route_distribution': route_distribution,
            'avg_response_time': round(analytics_data['avg_response_time'], 2),
            'avg_confidence': round(avg_confidence, 2),
            'error_count': analytics_data['error_count'],
            'active_sessions': len(analytics_data['user_sessions']),
            'recent_queries_count': len(recent_queries),
            'hourly_data': hourly_data,
            'daily_data': daily_data,
            'recent_queries': recent_queries[-10:] if recent_queries else []  # Last 10 queries
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = session.get('session_id')
    if session_id:
        # Join room for this session
        from flask_socketio import join_room
        join_room(session_id)
        emit('connected', {'session_id': session_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)