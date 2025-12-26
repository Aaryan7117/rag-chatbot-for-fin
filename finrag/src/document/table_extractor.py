"""
Table Extraction and Storage
============================

Tables are NEVER vectorized - they are extracted to structured JSON/SQLite.
Each table is linked to explanatory chunk_ids for context.

Why separate table handling:
- Tables contain structured numerical data
- Embeddings don't capture tabular relationships
- SQL enables precise numerical queries
- LLM can read structured tables better than flattened text
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

from .parser import DocumentElement, ElementType, ParsedDocument


class ColumnType(Enum):
    """Detected column types for schema inference."""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass
class TableColumn:
    """Column definition with inferred type."""
    name: str
    column_type: ColumnType
    original_index: int
    sample_values: List[str] = field(default_factory=list)


@dataclass
class ExtractedTable:
    """
    A fully extracted and structured table.
    
    Contains:
    - Structured data (rows and columns)
    - Inferred schema with types
    - Link to source chunk for context
    - Caption and surrounding text
    """
    
    table_id: str
    doc_id: str
    
    # Structure
    columns: List[TableColumn]
    rows: List[Dict[str, Any]]  # List of {column_name: value}
    
    # Source tracking
    page_number: int
    caption: Optional[str] = None
    context_chunk_id: Optional[str] = None  # Chunk that references this table
    
    # Raw data for fallback
    raw_data: List[List[str]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "table_id": self.table_id,
            "doc_id": self.doc_id,
            "columns": [
                {
                    "name": col.name,
                    "type": col.column_type.value,
                    "index": col.original_index
                }
                for col in self.columns
            ],
            "rows": self.rows,
            "page_number": self.page_number,
            "caption": self.caption,
            "context_chunk_id": self.context_chunk_id,
            "row_count": len(self.rows),
            "column_count": len(self.columns)
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.columns or not self.rows:
            return ""
        
        lines = []
        
        # Header
        header = "| " + " | ".join(col.name for col in self.columns) + " |"
        lines.append(header)
        
        # Separator
        separator = "|" + "|".join("---" for _ in self.columns) + "|"
        lines.append(separator)
        
        # Rows
        for row in self.rows:
            row_values = [str(row.get(col.name, "")) for col in self.columns]
            lines.append("| " + " | ".join(row_values) + " |")
        
        return "\n".join(lines)
    
    def to_sql_create(self, table_name: Optional[str] = None) -> str:
        """Generate SQL CREATE TABLE statement."""
        name = table_name or f"table_{self.table_id}"
        
        type_mapping = {
            ColumnType.TEXT: "TEXT",
            ColumnType.NUMBER: "REAL",
            ColumnType.CURRENCY: "REAL",
            ColumnType.PERCENTAGE: "REAL",
            ColumnType.DATE: "TEXT",
            ColumnType.UNKNOWN: "TEXT"
        }
        
        columns_sql = []
        for col in self.columns:
            sql_type = type_mapping[col.column_type]
            # Sanitize column name for SQL
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", col.name)
            columns_sql.append(f"    {safe_name} {sql_type}")
        
        return f"CREATE TABLE IF NOT EXISTS {name} (\n" + ",\n".join(columns_sql) + "\n);"


class TableExtractor:
    """
    Extracts and structures tables from parsed documents.
    
    Process:
    1. Get raw table data from parser
    2. Clean and normalize values
    3. Infer column types
    4. Link to surrounding context
    5. Output structured ExtractedTable
    """
    
    # Currency patterns
    CURRENCY_PATTERN = re.compile(r"^\$?\s*[\d,]+\.?\d*\s*(?:million|billion|M|B)?$", re.I)
    
    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(r"^[\d,]+\.?\d*\s*%$")
    
    # Number pattern
    NUMBER_PATTERN = re.compile(r"^[\d,]+\.?\d*$")
    
    # Date patterns
    DATE_PATTERNS = [
        re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # MM/DD/YYYY
        re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # YYYY-MM-DD
        re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.I)  # Month names
    ]
    
    def __init__(self):
        """Initialize table extractor."""
        pass
    
    def extract_tables(
        self, 
        document: ParsedDocument
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a parsed document.
        
        Args:
            document: ParsedDocument containing table elements
            
        Returns:
            List of ExtractedTable objects
        """
        extracted = []
        
        for table_element in document.tables:
            if table_element.element_type != ElementType.TABLE:
                continue
            
            raw_data = table_element.metadata.get("raw_data", [])
            if not raw_data:
                continue
            
            table = self._process_table(
                raw_data=raw_data,
                table_id=table_element.table_id,
                doc_id=document.doc_id,
                page_number=table_element.page_number
            )
            
            if table:
                extracted.append(table)
        
        return extracted
    
    def _process_table(
        self,
        raw_data: List[List],
        table_id: str,
        doc_id: str,
        page_number: int
    ) -> Optional[ExtractedTable]:
        """
        Process raw table data into structured format.
        
        Steps:
        1. Clean cell values
        2. Identify header row
        3. Infer column types
        4. Build structured rows
        """
        if not raw_data or len(raw_data) < 2:
            return None
        
        # Clean all cells
        cleaned = [
            [self._clean_cell(cell) for cell in row]
            for row in raw_data
        ]
        
        # First row is usually header
        header_row = cleaned[0]
        data_rows = cleaned[1:]
        
        # Generate column names
        columns = self._build_columns(header_row, data_rows)
        
        if not columns:
            return None
        
        # Build structured rows
        rows = []
        for data_row in data_rows:
            row_dict = {}
            for col in columns:
                if col.original_index < len(data_row):
                    value = self._parse_value(
                        data_row[col.original_index], 
                        col.column_type
                    )
                    row_dict[col.name] = value
                else:
                    row_dict[col.name] = None
            rows.append(row_dict)
        
        return ExtractedTable(
            table_id=table_id,
            doc_id=doc_id,
            columns=columns,
            rows=rows,
            page_number=page_number,
            raw_data=cleaned,
            metadata={
                "original_row_count": len(raw_data),
                "original_col_count": len(header_row) if header_row else 0
            }
        )
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean a table cell value."""
        if cell is None:
            return ""
        
        text = str(cell).strip()
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text
    
    def _build_columns(
        self, 
        header_row: List[str], 
        data_rows: List[List[str]]
    ) -> List[TableColumn]:
        """
        Build column definitions with inferred types.
        
        Infers types by analyzing data values in each column.
        """
        columns = []
        
        for i, header in enumerate(header_row):
            # Generate column name
            name = header if header else f"column_{i+1}"
            name = self._sanitize_column_name(name)
            
            # Collect sample values from data rows
            samples = []
            for row in data_rows[:10]:  # Sample first 10 rows
                if i < len(row) and row[i]:
                    samples.append(row[i])
            
            # Infer column type
            col_type = self._infer_column_type(samples)
            
            columns.append(TableColumn(
                name=name,
                column_type=col_type,
                original_index=i,
                sample_values=samples[:3]
            ))
        
        return columns
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for use in queries."""
        # Remove special characters
        clean = re.sub(r"[^a-zA-Z0-9_\s]", "", name)
        # Replace spaces with underscores
        clean = re.sub(r"\s+", "_", clean.strip())
        # Ensure starts with letter
        if clean and not clean[0].isalpha():
            clean = "col_" + clean
        return clean[:50] or "unnamed_column"
    
    def _infer_column_type(self, samples: List[str]) -> ColumnType:
        """
        Infer column type from sample values.
        
        Priority: Currency > Percentage > Number > Date > Text
        """
        if not samples:
            return ColumnType.UNKNOWN
        
        type_counts = {
            ColumnType.CURRENCY: 0,
            ColumnType.PERCENTAGE: 0,
            ColumnType.NUMBER: 0,
            ColumnType.DATE: 0,
            ColumnType.TEXT: 0
        }
        
        for sample in samples:
            if self.CURRENCY_PATTERN.match(sample):
                type_counts[ColumnType.CURRENCY] += 1
            elif self.PERCENTAGE_PATTERN.match(sample):
                type_counts[ColumnType.PERCENTAGE] += 1
            elif self.NUMBER_PATTERN.match(sample):
                type_counts[ColumnType.NUMBER] += 1
            elif any(p.match(sample) for p in self.DATE_PATTERNS):
                type_counts[ColumnType.DATE] += 1
            else:
                type_counts[ColumnType.TEXT] += 1
        
        # Return type with highest count (except TEXT as default)
        numeric_types = [ColumnType.CURRENCY, ColumnType.PERCENTAGE, ColumnType.NUMBER, ColumnType.DATE]
        for t in numeric_types:
            if type_counts[t] > len(samples) / 2:
                return t
        
        return ColumnType.TEXT
    
    def _parse_value(self, value: str, column_type: ColumnType) -> Any:
        """
        Parse string value to appropriate Python type.
        """
        if not value:
            return None
        
        if column_type in (ColumnType.NUMBER, ColumnType.CURRENCY, ColumnType.PERCENTAGE):
            # Remove non-numeric characters except decimal point
            numeric_str = re.sub(r"[^\d.]", "", value)
            try:
                return float(numeric_str) if numeric_str else None
            except ValueError:
                return value
        
        return value


def extract_tables(document: ParsedDocument) -> List[ExtractedTable]:
    """Convenience function to extract tables from a document."""
    extractor = TableExtractor()
    return extractor.extract_tables(document)
