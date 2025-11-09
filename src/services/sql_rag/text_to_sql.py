"""Text-to-SQL generation using LLMs for natural language queries over tables."""

from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
from anthropic import Anthropic

from src.core.config import settings
from src.core.logger import get_logger
from .sql_store import SQLStore

logger = get_logger(__name__)


class TextToSQLGenerator:
    """Generate SQL queries from natural language using LLMs.
    
    Features:
    - Schema-aware SQL generation
    - Support for aggregations, joins, and complex queries
    - Query validation and error handling
    - Multi-table reasoning
    """

    def __init__(
        self,
        sql_store: SQLStore,
        provider: str = "openai",  # openai or anthropic
    ):
        """Initialize Text-to-SQL generator.
        
        Args:
            sql_store: SQL store instance
            provider: LLM provider (openai or anthropic)
        """
        self.sql_store = sql_store
        self.provider = provider
        
        if provider == "openai":
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model = "claude-3-5-sonnet-20241022"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized TextToSQLGenerator with {provider}")

    def _get_relevant_schemas(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        max_tables: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get relevant table schemas for query.
        
        Args:
            query: Natural language query
            document_ids: Filter by document IDs
            max_tables: Maximum number of tables to include
            
        Returns:
            List of table schemas
        """
        # Get all table metadata
        if document_ids:
            all_metadata = []
            for doc_id in document_ids:
                all_metadata.extend(self.sql_store.get_table_metadata(document_id=doc_id))
        else:
            all_metadata = self.sql_store.get_table_metadata()
        
        # Simple relevance: check if query mentions caption or column names
        # In production, use embedding similarity
        scored_tables = []
        for meta in all_metadata:
            score = 0
            # Check caption
            if meta.get("caption"):
                caption_lower = meta["caption"].lower()
                query_lower = query.lower()
                if any(word in caption_lower for word in query_lower.split()):
                    score += 2
            
            # Check column names
            try:
                columns_info = eval(meta["columns_json"])  # Safe in controlled environment
                original_cols = columns_info.get("original", [])
                for col in original_cols:
                    if col.lower() in query.lower():
                        score += 1
            except:
                pass
            
            scored_tables.append((score, meta))
        
        # Sort by score and take top k
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        relevant_metadata = [meta for _, meta in scored_tables[:max_tables]]
        
        # Get full schemas
        schemas = []
        for meta in relevant_metadata:
            try:
                schema = self.sql_store.get_table_schema(meta["table_id"])
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"Could not get schema for {meta['table_id']}: {e}")
        
        return schemas

    def _build_schema_prompt(
        self,
        schemas: List[Dict[str, Any]],
    ) -> str:
        """Build schema description for prompt.
        
        Args:
            schemas: List of table schemas
            
        Returns:
            Formatted schema description
        """
        schema_parts = []
        
        for schema in schemas:
            table_name = schema["table_name"]
            meta = schema["metadata"]
            
            # Table description
            desc = f"\nTable: {table_name}"
            if meta.get("caption"):
                desc += f"\nCaption: {meta['caption']}"
            desc += f"\nDocument ID: {meta['document_id']}, Page: {meta['page_number']}"
            desc += f"\nRows: {meta['row_count']}, Columns: {meta['column_count']}"
            
            # Columns
            desc += "\nColumns:"
            for col in schema["columns"]:
                desc += f"\n  - {col['name']} ({col['type']})"
            
            schema_parts.append(desc)
        
        return "\n".join(schema_parts)

    def generate_sql(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language.
        
        Args:
            query: Natural language query
            document_ids: Filter by document IDs
            
        Returns:
            Dictionary with generated SQL and metadata
        """
        # Get relevant schemas
        schemas = self._get_relevant_schemas(query, document_ids)
        
        if not schemas:
            return {
                "success": False,
                "error": "No relevant tables found for query",
                "sql": None,
            }
        
        # Build schema prompt
        schema_prompt = self._build_schema_prompt(schemas)
        
        # Build full prompt
        system_prompt = """You are an expert SQL query generator. Given table schemas and a natural language query, generate a valid SQL query.

Rules:
1. Only use tables and columns from the provided schemas
2. Generate syntactically correct SQL for the database
3. Use appropriate JOINs when querying multiple tables
4. Include necessary WHERE clauses and GROUP BY for aggregations
5. Return ONLY the SQL query, no explanations
6. If the query cannot be answered with available tables, return: NO_TABLES_MATCH
"""
        
        user_prompt = f"""Available table schemas:
{schema_prompt}

Natural language query: {query}

Generate SQL query:"""
        
        # Generate SQL
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                sql_query = response.choices[0].message.content.strip()
            else:  # anthropic
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                sql_query = response.content[0].text.strip()
            
            # Clean up SQL
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            if "NO_TABLES_MATCH" in sql_query:
                return {
                    "success": False,
                    "error": "Query cannot be answered with available tables",
                    "sql": None,
                    "schemas_used": schemas,
                }
            
            return {
                "success": True,
                "sql": sql_query,
                "schemas_used": schemas,
                "query": query,
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": None,
            }

    def execute_natural_language_query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        return_dataframe: bool = True,
    ) -> Dict[str, Any]:
        """Execute natural language query end-to-end.
        
        Args:
            query: Natural language query
            document_ids: Filter by document IDs
            return_dataframe: If True, return DataFrame; else list of dicts
            
        Returns:
            Dictionary with query results and metadata
        """
        # Generate SQL
        sql_result = self.generate_sql(query, document_ids)
        
        if not sql_result["success"]:
            return sql_result
        
        sql_query = sql_result["sql"]
        
        # Execute SQL
        try:
            results = self.sql_store.execute_query(sql_query, return_dataframe)
            
            return {
                "success": True,
                "query": query,
                "sql": sql_query,
                "results": results,
                "row_count": len(results),
                "schemas_used": sql_result["schemas_used"],
            }
            
        except Exception as e:
            logger.error(f"Error executing generated SQL: {e}")
            return {
                "success": False,
                "error": f"SQL execution failed: {str(e)}",
                "query": query,
                "sql": sql_query,
                "results": None,
            }

    def validate_sql(
        self,
        sql: str,
    ) -> Dict[str, Any]:
        """Validate SQL query without executing.
        
        Args:
            sql: SQL query string
            
        Returns:
            Validation result
        """
        try:
            # Try to explain the query (doesn't execute)
            with self.sql_store.engine.connect() as conn:
                conn.execute(f"EXPLAIN {sql}")
            
            return {
                "valid": True,
                "sql": sql,
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "sql": sql,
            }
