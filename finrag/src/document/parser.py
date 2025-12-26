"""
PDF Parser with Structure-Aware Extraction
============================================

This module handles PDF parsing with emphasis on preserving document structure,
which is critical for SEC 10-K filings and financial reports.

Design Decisions:
- Use pdfplumber for table detection and text extraction
- Use PyMuPDF (fitz) for faster text extraction when structure isn't critical
- Track page numbers for every extracted element (required for citations)
- Detect section headers for structural chunking
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import pdfplumber
import fitz  # PyMuPDF


class ElementType(Enum):
    """Types of document elements."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class DocumentElement:
    """
    A single element from the parsed document.
    
    Every element tracks its source location for citation purposes.
    """
    element_type: ElementType
    content: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    heading_level: Optional[int] = None  # 1-6 for headings
    table_id: Optional[str] = None  # For table elements
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """
    A document section containing multiple elements.
    
    Sections are defined by headings and contain all content
    until the next section of equal or higher level.
    """
    section_id: str
    title: str
    level: int  # Heading level (1 = top-level)
    item_number: Optional[str] = None  # SEC Item number if applicable
    page_start: int = 0
    page_end: int = 0
    elements: List[DocumentElement] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Complete parsed document with structure.
    
    Contains all text, tables, and structural information
    needed for the chunking and indexing pipeline.
    """
    doc_id: str
    filename: str
    title: Optional[str] = None
    total_pages: int = 0
    sections: List[Section] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)  # Flat list
    tables: List[DocumentElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_text(self) -> str:
        """Get all text content concatenated."""
        return "\n\n".join(
            el.content for el in self.elements 
            if el.element_type != ElementType.TABLE
        )


class PDFParser:
    """
    Structure-aware PDF parser for financial documents.
    
    Extraction Strategy:
    1. Extract raw text with page tracking
    2. Detect and extract tables separately
    3. Identify section headers (SEC Items, Parts, etc.)
    4. Build hierarchical document structure
    
    Why this approach:
    - SEC 10-K filings have predictable structure (Items 1-15)
    - Tables contain critical numerical data that must be preserved
    - Section structure enables coherent chunking
    """
    
    # SEC 10-K Item patterns - these define major sections
    SEC_ITEM_PATTERN = re.compile(
        r"(?i)^[\s]*ITEM\s+(\d+[A-Z]?)[\.\s]+(.+?)(?:\n|$)",
        re.MULTILINE
    )
    
    # Part patterns (PART I, PART II, etc.)
    SEC_PART_PATTERN = re.compile(
        r"(?i)^[\s]*PART\s+([IVX]+)[\.\s]*(.*)(?:\n|$)",
        re.MULTILINE
    )
    
    # Generic heading patterns (all caps, short lines)
    HEADING_PATTERN = re.compile(
        r"^[\s]*([A-Z][A-Z\s,&\-]{5,50})[\s]*$",
        re.MULTILINE
    )
    
    def __init__(self, min_table_rows: int = 2, min_table_cols: int = 2):
        """
        Initialize parser.
        
        Args:
            min_table_rows: Minimum rows to consider as table
            min_table_cols: Minimum columns to consider as table
        """
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
    
    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF document with full structure extraction.
        
        This is the main entry point for document processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with all extracted content and structure
        """
        pdf_path = Path(pdf_path)
        
        # Generate stable document ID from file hash
        doc_id = self._generate_doc_id(pdf_path)
        
        # Initialize document
        doc = ParsedDocument(
            doc_id=doc_id,
            filename=pdf_path.name,
            metadata={
                "source_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
            }
        )
        
        # Extract all content with both libraries
        elements, tables = self._extract_with_pdfplumber(pdf_path)
        doc.elements = elements
        doc.tables = tables
        doc.total_pages = self._get_page_count(pdf_path)
        
        # Build section structure
        doc.sections = self._build_sections(elements)
        
        # Extract document title (usually first heading)
        doc.title = self._extract_title(elements)
        
        return doc
    
    def _generate_doc_id(self, pdf_path: Path) -> str:
        """
        Generate stable document ID from file content hash.
        
        Why hash-based: Ensures same document always gets same ID,
        enabling idempotent ingestion and deduplication.
        """
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get total page count using PyMuPDF (faster)."""
        with fitz.open(pdf_path) as doc:
            return len(doc)
    
    def _extract_with_pdfplumber(
        self, pdf_path: Path
    ) -> Tuple[List[DocumentElement], List[DocumentElement]]:
        """
        Extract text and tables using pdfplumber.
        
        Why pdfplumber:
        - Excellent table detection with bbox tracking
        - Preserves spatial layout information
        - Good handling of multi-column layouts
        """
        elements = []
        tables = []
        table_counter = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables first (to exclude from text extraction)
                page_tables = page.extract_tables()
                table_bboxes = []
                
                for table_data in page_tables:
                    if self._is_valid_table(table_data):
                        table_counter += 1
                        table_id = f"table_{table_counter:04d}"
                        
                        # Get table bounding box
                        table_settings = page.find_tables()
                        bbox = None
                        if table_settings:
                            for t in table_settings:
                                bbox = t.bbox
                                table_bboxes.append(bbox)
                                break
                        
                        table_element = DocumentElement(
                            element_type=ElementType.TABLE,
                            content=self._table_to_string(table_data),
                            page_number=page_num,
                            bbox=bbox,
                            table_id=table_id,
                            metadata={"raw_data": table_data}
                        )
                        tables.append(table_element)
                
                # Extract text, excluding table regions
                text = page.extract_text() or ""
                
                # Parse text into elements
                page_elements = self._parse_text_to_elements(text, page_num)
                elements.extend(page_elements)
        
        return elements, tables
    
    def _is_valid_table(self, table_data: List[List]) -> bool:
        """Check if extracted data qualifies as a table."""
        if not table_data:
            return False
        
        rows = len(table_data)
        cols = max(len(row) for row in table_data) if table_data else 0
        
        return rows >= self.min_table_rows and cols >= self.min_table_cols
    
    def _table_to_string(self, table_data: List[List]) -> str:
        """Convert table data to markdown-style string."""
        if not table_data:
            return ""
        
        lines = []
        for i, row in enumerate(table_data):
            # Clean cell values
            cells = [str(cell).strip() if cell else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
            
            # Add header separator after first row
            if i == 0:
                lines.append("|" + "|".join(["---"] * len(cells)) + "|")
        
        return "\n".join(lines)
    
    def _parse_text_to_elements(
        self, text: str, page_number: int
    ) -> List[DocumentElement]:
        """
        Parse raw text into structured elements.
        
        Identifies:
        - SEC Item headers
        - Part headers
        - Generic headings (ALL CAPS)
        - Regular paragraphs
        """
        elements = []
        
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            element = self._classify_paragraph(para, page_number)
            elements.append(element)
        
        return elements
    
    def _classify_paragraph(
        self, text: str, page_number: int
    ) -> DocumentElement:
        """Classify a paragraph as heading, list, or regular text."""
        
        # Check for SEC Item header
        item_match = self.SEC_ITEM_PATTERN.match(text)
        if item_match:
            return DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page_number=page_number,
                heading_level=2,  # Items are level 2
                metadata={
                    "item_number": item_match.group(1),
                    "item_title": item_match.group(2).strip()
                }
            )
        
        # Check for Part header
        part_match = self.SEC_PART_PATTERN.match(text)
        if part_match:
            return DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page_number=page_number,
                heading_level=1,  # Parts are level 1
                metadata={
                    "part_number": part_match.group(1),
                    "part_title": part_match.group(2).strip()
                }
            )
        
        # Check for generic heading (ALL CAPS, short)
        if self._is_heading(text):
            return DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page_number=page_number,
                heading_level=3  # Generic headings are level 3
            )
        
        # Check for list
        if self._is_list(text):
            return DocumentElement(
                element_type=ElementType.LIST,
                content=text,
                page_number=page_number
            )
        
        # Default to paragraph
        return DocumentElement(
            element_type=ElementType.PARAGRAPH,
            content=text,
            page_number=page_number
        )
    
    def _is_heading(self, text: str) -> bool:
        """Check if text is likely a heading."""
        # Short, mostly uppercase
        if len(text) > 100:
            return False
        
        # Count uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return False
        
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        
        # Heading if >70% uppercase and not too long
        return upper_ratio > 0.7 and len(text) < 80
    
    def _is_list(self, text: str) -> bool:
        """Check if text is a list item or bullet points."""
        list_patterns = [
            r"^\s*[\u2022\u2023\u25E6\u2043\u2219]\s",  # Bullet chars
            r"^\s*[-\*]\s",  # Markdown bullets
            r"^\s*\d+[\.\)]\s",  # Numbered list
            r"^\s*[a-z][\.\)]\s",  # Lettered list
        ]
        
        return any(re.match(p, text, re.MULTILINE) for p in list_patterns)
    
    def _build_sections(
        self, elements: List[DocumentElement]
    ) -> List[Section]:
        """
        Build hierarchical section structure from flat elements.
        
        This creates a tree structure based on heading levels,
        which is critical for context-aware chunking.
        """
        sections = []
        section_stack = []  # Stack of (level, section) tuples
        
        for element in elements:
            if element.element_type == ElementType.HEADING:
                level = element.heading_level or 3
                
                # Create new section
                section = Section(
                    section_id=self._generate_section_id(element.content),
                    title=element.content,
                    level=level,
                    item_number=element.metadata.get("item_number"),
                    page_start=element.page_number
                )
                
                # Pop sections of equal or lower level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                
                # Add to parent or root
                if section_stack:
                    section_stack[-1][1].subsections.append(section)
                else:
                    sections.append(section)
                
                section_stack.append((level, section))
            
            elif section_stack:
                # Add element to current section
                current_section = section_stack[-1][1]
                current_section.elements.append(element)
                current_section.page_end = element.page_number
        
        return sections
    
    def _generate_section_id(self, title: str) -> str:
        """Generate stable section ID from title."""
        # Clean and hash
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
        clean = re.sub(r"\s+", "_", clean.strip())[:50]
        
        # Add short hash for uniqueness
        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"{clean}_{hash_suffix}"
    
    def _extract_title(self, elements: List[DocumentElement]) -> Optional[str]:
        """Extract document title from first heading."""
        for element in elements[:10]:  # Check first 10 elements
            if element.element_type == ElementType.HEADING:
                return element.content.strip()
        return None


# Convenience function for simple usage
def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """Parse a PDF file and return structured document."""
    parser = PDFParser()
    return parser.parse(pdf_path)
