"""SQL store for structured data from tables and forms."""

import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, Text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import hashlib

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class SQLStore:
    """PostgreSQL/DuckDB store for structured document data.
    
    Features:
    - Store table data with schema inference
    - Support for multiple tables per document
    - Query execution with result formatting
    - Metadata tracking for table provenance
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        use_duckdb: bool = True,
    ):
        """Initialize SQL store.
        
        Args:
            database_url: Database connection URL
            use_duckdb: If True, use DuckDB (faster for analytics), else PostgreSQL
        """
        if database_url:
            self.database_url = database_url
        elif use_duckdb:
            # DuckDB for fast analytical queries
            self.database_url = "duckdb:///data/multimodal_rag.duckdb"
        else:
            # PostgreSQL for production
            self.database_url = settings.POSTGRES_URL
        
        self.engine = create_engine(self.database_url)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)
        
        # Create metadata table
        self._create_metadata_table()
        
        logger.info(f"Initialized SQLStore with {self.database_url}")

    def _create_metadata_table(self):
        """Create table for tracking document table metadata."""
        metadata_table = Table(
            "document_tables_metadata",
            self.metadata,
            Column("table_id", String, primary_key=True),
            Column("document_id", String, nullable=False),
            Column("table_name", String, nullable=False),
            Column("page_number", Integer, nullable=False),
            Column("bbox", Text),  # JSON string of bounding box
            Column("caption", Text),
            Column("row_count", Integer),
            Column("column_count", Integer),
            Column("columns_json", Text),  # JSON string of column names and types
            Column("created_at", String),
            extend_existing=True,
        )
        
        self.metadata.create_all(self.engine)
        logger.info("Created metadata table")

    def _generate_table_id(self, document_id: str, page_number: int, table_index: int) -> str:
        """Generate unique table ID."""
        raw = f"{document_id}_{page_number}_{table_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _sanitize_table_name(self, document_id: str, table_id: str) -> str:
        """Generate safe SQL table name."""
        # Remove special characters and limit length
        safe_doc_id = "".join(c if c.isalnum() else "_" for c in document_id)[:20]
        return f"doc_{safe_doc_id}_table_{table_id}"

    def _sanitize_column_name(self, name: str) -> str:
        """Generate safe SQL column name."""
        # Remove special characters, replace spaces with underscore
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        # Ensure starts with letter
        if safe_name and not safe_name[0].isalpha():
            safe_name = "col_" + safe_name
        return safe_name.lower()[:63]  # PostgreSQL limit

    def store_table(
        self,
        document_id: str,
        table_data: pd.DataFrame,
        page_number: int,
        table_index: int = 0,
        caption: Optional[str] = None,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """Store table data in SQL database.
        
        Args:
            document_id: Source document ID
            table_data: Pandas DataFrame with table data
            page_number: Page number where table appears
            table_index: Index of table on page (for multiple tables)
            caption: Table caption or title
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            table_id: Unique identifier for stored table
        """
        try:
            # Generate IDs and names
            table_id = self._generate_table_id(document_id, page_number, table_index)
            table_name = self._sanitize_table_name(document_id, table_id)
            
            # Sanitize column names
            table_data = table_data.copy()
            column_mapping = {}
            for col in table_data.columns:
                safe_col = self._sanitize_column_name(str(col))
                column_mapping[col] = safe_col
            table_data.rename(columns=column_mapping, inplace=True)
            
            # Store table data
            table_data.to_sql(
                table_name,
                self.engine,
                if_exists="replace",
                index=False,
            )
            
            # Store metadata
            columns_info = {
                "original": list(column_mapping.keys()),
                "sanitized": list(column_mapping.values()),
                "types": {col: str(dtype) for col, dtype in table_data.dtypes.items()},
            }
            
            metadata_row = {
                "table_id": table_id,
                "document_id": document_id,
                "table_name": table_name,
                "page_number": page_number,
                "bbox": str(bbox) if bbox else None,
                "caption": caption,
                "row_count": len(table_data),
                "column_count": len(table_data.columns),
                "columns_json": str(columns_info),
                "created_at": pd.Timestamp.now().isoformat(),
            }
            
            metadata_df = pd.DataFrame([metadata_row])
            metadata_df.to_sql(
                "document_tables_metadata",
                self.engine,
                if_exists="append",
                index=False,
            )
            
            logger.info(f"Stored table {table_id} with {len(table_data)} rows, {len(table_data.columns)} columns")
            return table_id
            
        except Exception as e:
            logger.error(f"Error storing table: {e}")
            raise

    def execute_query(
        self,
        query: str,
        return_dataframe: bool = True,
    ) -> pd.DataFrame | List[Dict[str, Any]]:
        """Execute SQL query and return results.
        
        Args:
            query: SQL query string
            return_dataframe: If True, return DataFrame; else list of dicts
            
        Returns:
            Query results as DataFrame or list of dicts
        """
        try:
            if return_dataframe:
                result = pd.read_sql(query, self.engine)
            else:
                with self.engine.connect() as conn:
                    result_proxy = conn.execute(query)
                    result = [dict(row) for row in result_proxy]
            
            logger.info(f"Executed query, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_table_metadata(
        self,
        document_id: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metadata for tables.
        
        Args:
            document_id: Filter by document ID
            table_id: Filter by specific table ID
            
        Returns:
            List of metadata records
        """
        query = "SELECT * FROM document_tables_metadata WHERE 1=1"
        
        if document_id:
            query += f" AND document_id = '{document_id}'"
        if table_id:
            query += f" AND table_id = '{table_id}'"
        
        return self.execute_query(query, return_dataframe=False)

    def get_table_data(
        self,
        table_id: str,
    ) -> pd.DataFrame:
        """Retrieve table data by ID.
        
        Args:
            table_id: Table identifier
            
        Returns:
            DataFrame with table data
        """
        # Get table name from metadata
        metadata = self.get_table_metadata(table_id=table_id)
        if not metadata:
            raise ValueError(f"Table {table_id} not found")
        
        table_name = metadata[0]["table_name"]
        
        # Query table
        query = f"SELECT * FROM {table_name}"
        return self.execute_query(query, return_dataframe=True)

    def get_table_schema(
        self,
        table_id: str,
    ) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_id: Table identifier
            
        Returns:
            Dictionary with schema information
        """
        metadata = self.get_table_metadata(table_id=table_id)
        if not metadata:
            raise ValueError(f"Table {table_id} not found")
        
        table_name = metadata[0]["table_name"]
        
        # Get schema from database
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        
        return {
            "table_id": table_id,
            "table_name": table_name,
            "columns": columns,
            "metadata": metadata[0],
        }

    def list_tables(
        self,
        document_id: Optional[str] = None,
    ) -> List[str]:
        """List all table IDs, optionally filtered by document.
        
        Args:
            document_id: Optional document ID filter
            
        Returns:
            List of table IDs
        """
        metadata = self.get_table_metadata(document_id=document_id)
        return [m["table_id"] for m in metadata]

    def delete_document_tables(
        self,
        document_id: str,
    ) -> int:
        """Delete all tables for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of tables deleted
        """
        # Get table names
        metadata = self.get_table_metadata(document_id=document_id)
        
        count = 0
        for meta in metadata:
            table_name = meta["table_name"]
            # Drop table
            with self.engine.connect() as conn:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            count += 1
        
        # Delete metadata
        with self.engine.connect() as conn:
            conn.execute(
                f"DELETE FROM document_tables_metadata WHERE document_id = '{document_id}'"
            )
        
        logger.info(f"Deleted {count} tables for document {document_id}")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Count tables
        metadata = self.get_table_metadata()
        
        return {
            "total_tables": len(metadata),
            "unique_documents": len(set(m["document_id"] for m in metadata)),
            "total_rows": sum(m["row_count"] for m in metadata),
            "tables": metadata,
        }
