"""
SQLite Table Store
==================

Structured storage for extracted tables. Tables are NEVER vectorized.

This store enables:
- Precise numerical queries on table data
- Schema-aware table lookups
- Table content retrieval by table_id
- Linking tables to explanatory chunks
"""

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.document.table_extractor import ExtractedTable


class TableStore:
    """
    SQLite-based storage for extracted tables.
    
    Tables are stored in normalized form:
    - table_metadata: Table info and schema
    - table_data: Actual table content as JSON
    
    Linked to chunks via context_chunk_id for retrieval.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize table store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.paths.tables_dir / settings.stores.sqlite_db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_metadata (
                    table_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    caption TEXT,
                    page_number INTEGER,
                    column_count INTEGER,
                    row_count INTEGER,
                    context_chunk_id TEXT,
                    schema_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_data (
                    table_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    PRIMARY KEY (table_id, row_index),
                    FOREIGN KEY (table_id) REFERENCES table_metadata(table_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_doc 
                ON table_metadata(doc_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_chunk 
                ON table_metadata(context_chunk_id)
            """)
    
    def add_table(self, table: ExtractedTable) -> None:
        """
        Store an extracted table.
        
        Args:
            table: ExtractedTable object from table_extractor
        """
        with self._get_connection() as conn:
            # Store metadata
            schema = [
                {"name": col.name, "type": col.column_type.value}
                for col in table.columns
            ]
            
            conn.execute("""
                INSERT OR REPLACE INTO table_metadata
                (table_id, doc_id, caption, page_number, column_count,
                 row_count, context_chunk_id, schema_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                table.table_id,
                table.doc_id,
                table.caption,
                table.page_number,
                len(table.columns),
                len(table.rows),
                table.context_chunk_id,
                json.dumps(schema)
            ))
            
            # Store row data
            conn.execute("DELETE FROM table_data WHERE table_id = ?", (table.table_id,))
            
            for i, row in enumerate(table.rows):
                conn.execute("""
                    INSERT INTO table_data (table_id, row_index, row_json)
                    VALUES (?, ?, ?)
                """, (table.table_id, i, json.dumps(row)))
    
    def add_tables(self, tables: List[ExtractedTable]) -> int:
        """Add multiple tables in a batch."""
        for table in tables:
            self.add_table(table)
        return len(tables)
    
    def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a table by its ID.
        
        Returns:
            Dictionary with table metadata and rows
        """
        with self._get_connection() as conn:
            # Get metadata
            cursor = conn.execute("""
                SELECT * FROM table_metadata WHERE table_id = ?
            """, (table_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            table = dict(row)
            table['schema'] = json.loads(table.get('schema_json', '[]'))
            
            # Get rows
            cursor = conn.execute("""
                SELECT row_json FROM table_data 
                WHERE table_id = ? 
                ORDER BY row_index
            """, (table_id,))
            
            table['rows'] = [json.loads(r['row_json']) for r in cursor.fetchall()]
            
            return table
    
    def get_tables_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get all tables referenced by a chunk."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT table_id FROM table_metadata 
                WHERE context_chunk_id = ?
            """, (chunk_id,))
            
            tables = []
            for row in cursor.fetchall():
                table = self.get_table(row['table_id'])
                if table:
                    tables.append(table)
            
            return tables
    
    def get_tables_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all tables from a document."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT table_id FROM table_metadata 
                WHERE doc_id = ?
                ORDER BY page_number
            """, (doc_id,))
            
            tables = []
            for row in cursor.fetchall():
                table = self.get_table(row['table_id'])
                if table:
                    tables.append(table)
            
            return tables
    
    def link_table_to_chunk(self, table_id: str, chunk_id: str) -> None:
        """Link a table to its explanatory chunk."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE table_metadata 
                SET context_chunk_id = ?
                WHERE table_id = ?
            """, (chunk_id, table_id))
    
    def format_table_for_llm(self, table_id: str) -> Optional[str]:
        """
        Format table as markdown for LLM consumption.
        
        Returns table in a format optimized for LLM understanding.
        """
        table = self.get_table(table_id)
        if not table:
            return None
        
        lines = []
        
        # Caption
        if table.get('caption'):
            lines.append(f"**{table['caption']}**")
            lines.append("")
        
        schema = table.get('schema', [])
        rows = table.get('rows', [])
        
        if not schema or not rows:
            return None
        
        # Header
        headers = [col['name'] for col in schema]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Rows (limit to avoid context overflow)
        for row in rows[:50]:  # Max 50 rows
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")
        
        if len(rows) > 50:
            lines.append(f"\n*[{len(rows) - 50} more rows not shown]*")
        
        return "\n".join(lines)
    
    def search_tables(
        self, 
        query: str, 
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search tables by caption or column names.
        
        Simple text search - for complex queries, use SQL directly.
        """
        with self._get_connection() as conn:
            sql = """
                SELECT table_id, caption, schema_json 
                FROM table_metadata 
                WHERE (caption LIKE ? OR schema_json LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%"]
            
            if doc_id:
                sql += " AND doc_id = ?"
                params.append(doc_id)
            
            cursor = conn.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                table = self.get_table(row['table_id'])
                if table:
                    results.append(table)
            
            return results
    
    def clear(self) -> None:
        """Clear all table data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM table_data")
            conn.execute("DELETE FROM table_metadata")
        print("[TableStore] Cleared all tables")
    
    @property
    def count(self) -> int:
        """Get number of stored tables."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM table_metadata")
            return cursor.fetchone()[0]
