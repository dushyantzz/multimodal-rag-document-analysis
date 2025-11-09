"""LangGraph-based agentic RAG system for intelligent query orchestration."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import numpy as np

from src.core.config import settings
from src.core.logger import get_logger
from .tools import AgentTools

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State for agent workflow."""
    
    # Input
    query: str
    document_ids: Optional[List[str]]
    include_images: bool
    conversation_history: List[Dict[str, str]]
    
    # Query analysis
    query_type: str  # semantic, numerical, hybrid, visual
    requires_sql: bool
    requires_visual: bool
    requires_text: bool
    
    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    sql_results: Optional[Dict[str, Any]]
    
    # Response generation
    response: str
    cited_sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    confidence: float
    
    # Control flow
    messages: Annotated[List, operator.add]
    next_step: str


class QueryAgent:
    """Agentic RAG system with multi-step reasoning.
    
    Workflow:
    1. Query Analysis: Classify query type and intent
    2. Route Selection: Choose retrieval strategy
    3. Retrieval: Execute multi-modal search and/or SQL
    4. Reranking: Cross-encoder reranking for precision
    5. Response Generation: Generate answer with citations
    6. Visual Grounding: Add relevant images and charts
    """

    def __init__(
        self,
        tools: AgentTools,
        llm_provider: str = "openai",
    ):
        """Initialize query agent.
        
        Args:
            tools: Agent tools instance
            llm_provider: LLM provider (openai or anthropic)
        """
        self.tools = tools
        self.llm_provider = llm_provider
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
            )
            self.vision_llm = ChatOpenAI(
                model="gpt-4-vision-preview",
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
            )
        else:
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0,
                api_key=settings.ANTHROPIC_API_KEY,
            )
            self.vision_llm = self.llm  # Claude 3.5 has vision
        
        # Build agent graph
        self.graph = self._build_graph()
        
        logger.info(f"Initialized QueryAgent with {llm_provider}")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("retrieve_semantic", self._retrieve_semantic)
        workflow.add_node("retrieve_sql", self._retrieve_sql)
        workflow.add_node("retrieve_hybrid", self._retrieve_hybrid)
        workflow.add_node("rerank", self._rerank_results)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("add_visuals", self._add_visuals)
        
        # Define edges
        workflow.set_entry_point("analyze_query")
        
        # From analyze -> route
        workflow.add_edge("analyze_query", "route_query")
        
        # From route -> retrieval (conditional)
        workflow.add_conditional_edges(
            "route_query",
            self._route_decision,
            {
                "semantic": "retrieve_semantic",
                "sql": "retrieve_sql",
                "hybrid": "retrieve_hybrid",
            },
        )
        
        # From retrieval -> rerank
        workflow.add_edge("retrieve_semantic", "rerank")
        workflow.add_edge("retrieve_sql", "rerank")
        workflow.add_edge("retrieve_hybrid", "rerank")
        
        # From rerank -> generate
        workflow.add_edge("rerank", "generate_response")
        
        # From generate -> visuals (conditional)
        workflow.add_conditional_edges(
            "generate_response",
            lambda state: "add_visuals" if state["include_images"] else "end",
            {
                "add_visuals": "add_visuals",
                "end": END,
            },
        )
        
        # From visuals -> end
        workflow.add_edge("add_visuals", END)
        
        return workflow.compile()

    def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyze query to determine intent and type."""
        query = state["query"]
        
        system_prompt = """Analyze the query and classify it into one of these types:

1. semantic: General question about document content, concepts, or explanations
2. numerical: Questions requiring calculations, aggregations, or statistical operations
3. visual: Questions about images, charts, diagrams, or visual elements
4. hybrid: Questions requiring both semantic understanding and numerical analysis

Also determine:
- requires_sql: True if query needs database operations (SUM, COUNT, AVG, etc.)
- requires_visual: True if query references visual elements
- requires_text: True if query needs text semantic search

Respond in JSON format:
{
  "query_type": "semantic|numerical|visual|hybrid",
  "requires_sql": boolean,
  "requires_visual": boolean,
  "requires_text": boolean,
  "reasoning": "Brief explanation"
}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}"),
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse response
        try:
            import json
            analysis = json.loads(response.content)
            
            state["query_type"] = analysis["query_type"]
            state["requires_sql"] = analysis["requires_sql"]
            state["requires_visual"] = analysis["requires_visual"]
            state["requires_text"] = analysis["requires_text"]
            state["messages"] = [AIMessage(content=analysis["reasoning"])]
            
            logger.info(f"Query analysis: {analysis['query_type']} - {analysis['reasoning']}")
            
        except Exception as e:
            logger.warning(f"Error parsing query analysis: {e}, using defaults")
            state["query_type"] = "semantic"
            state["requires_sql"] = False
            state["requires_visual"] = False
            state["requires_text"] = True
            state["messages"] = []
        
        return state

    def _route_decision(self, state: AgentState) -> str:
        """Decide which retrieval path to take."""
        if state["query_type"] == "numerical" and state["requires_sql"]:
            return "sql"
        elif state["query_type"] == "hybrid":
            return "hybrid"
        else:
            return "semantic"

    def _retrieve_semantic(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve using semantic search (text + visual embeddings)."""
        query = state["query"]
        document_ids = state.get("document_ids")
        
        # Generate embeddings
        text_embedding = self.tools.embed_text(query)
        visual_embedding = None
        if state["requires_visual"]:
            # For text queries about images, use text embedding
            # In production, convert query to image if user provides one
            visual_embedding = text_embedding  # Placeholder
        
        # Hybrid search
        results = self.tools.vector_store.hybrid_search(
            visual_embedding=visual_embedding,
            text_embedding=text_embedding,
            limit=10,
            visual_weight=0.6 if state["requires_visual"] else 0.3,
            document_ids=document_ids,
        )
        
        state["retrieved_chunks"] = [r.dict() for r in results]
        logger.info(f"Semantic retrieval: {len(results)} results")
        
        return state

    def _retrieve_sql(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve using SQL for numerical queries."""
        query = state["query"]
        document_ids = state.get("document_ids")
        
        # Execute natural language query
        sql_results = self.tools.text_to_sql.execute_natural_language_query(
            query=query,
            document_ids=document_ids,
            return_dataframe=False,
        )
        
        state["sql_results"] = sql_results
        state["retrieved_chunks"] = []
        
        logger.info(f"SQL retrieval: {sql_results.get('row_count', 0)} rows")
        
        return state

    def _retrieve_hybrid(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve using both semantic and SQL."""
        # Semantic retrieval
        state = self._retrieve_semantic(state)
        
        # SQL retrieval if needed
        if state["requires_sql"]:
            sql_state = self._retrieve_sql(state)
            state["sql_results"] = sql_state["sql_results"]
        
        logger.info("Hybrid retrieval completed")
        return state

    def _rerank_results(self, state: AgentState) -> Dict[str, Any]:
        """Rerank retrieved results using cross-encoder.
        
        TODO: Implement cross-encoder reranking for better precision.
        For now, keep top results as-is.
        """
        # Sort by score and keep top 5
        chunks = state["retrieved_chunks"]
        if chunks:
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            state["retrieved_chunks"] = chunks[:5]
        
        logger.info(f"Reranking: keeping top {len(state['retrieved_chunks'])} results")
        return state

    def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate final response with citations."""
        query = state["query"]
        chunks = state["retrieved_chunks"]
        sql_results = state.get("sql_results")
        
        # Build context
        context_parts = []
        
        # Add semantic context
        if chunks:
            context_parts.append("Relevant document excerpts:\n")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"[{i}] (Page {chunk['page_number']}, {chunk['chunk_type']}):\n{chunk['content']}\n"
                )
        
        # Add SQL results
        if sql_results and sql_results.get("success"):
            context_parts.append("\nNumerical data from tables:\n")
            context_parts.append(f"Query: {sql_results['sql']}\n")
            context_parts.append(f"Results: {sql_results['results']}\n")
        
        context = "\n".join(context_parts)
        
        # Generate response
        system_prompt = """You are an expert document analysis assistant. Answer the user's question based on the provided context.

Rules:
1. Cite sources using [1], [2], etc. format
2. For numerical data, reference the SQL results
3. Be concise but comprehensive
4. If information is insufficient, say so
5. Maintain accuracy - don't hallucinate facts
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
        ]
        
        response = self.vision_llm.invoke(messages)
        
        state["response"] = response.content
        state["cited_sources"] = chunks
        state["confidence"] = 0.85  # Placeholder
        
        logger.info("Generated response with citations")
        
        return state

    def _add_visuals(self, state: AgentState) -> Dict[str, Any]:
        """Add relevant images and visual grounding to response."""
        chunks = state["retrieved_chunks"]
        
        # Extract image chunks
        image_chunks = [
            chunk for chunk in chunks
            if chunk["chunk_type"] in ["image", "figure", "chart"]
        ]
        
        state["images"] = image_chunks[:3]  # Top 3 images
        
        logger.info(f"Added {len(state['images'])} visual elements")
        
        return state

    def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        include_images: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute agent query workflow.
        
        Args:
            query: User query
            document_ids: Optional document filters
            include_images: Whether to include visual elements
            conversation_history: Previous conversation turns
            
        Returns:
            Response dictionary with answer, sources, and visuals
        """
        # Initialize state
        initial_state = AgentState(
            query=query,
            document_ids=document_ids,
            include_images=include_images,
            conversation_history=conversation_history or [],
            query_type="",
            requires_sql=False,
            requires_visual=False,
            requires_text=True,
            retrieved_chunks=[],
            sql_results=None,
            response="",
            cited_sources=[],
            images=[],
            confidence=0.0,
            messages=[],
            next_step="",
        )
        
        # Execute workflow
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "success": True,
                "query": query,
                "answer": final_state["response"],
                "query_type": final_state["query_type"],
                "sources": final_state["cited_sources"],
                "images": final_state["images"],
                "sql_results": final_state.get("sql_results"),
                "confidence": final_state["confidence"],
            }
            
        except Exception as e:
            logger.error(f"Error executing agent workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
            }
