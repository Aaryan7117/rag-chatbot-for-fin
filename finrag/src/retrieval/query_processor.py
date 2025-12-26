"""
Query Processor - Normalization and Expansion
==============================================

Prepares user queries for retrieval by:
1. Normalizing text (lowercase, clean)
2. Detecting query intent
3. Expanding with synonyms and domain terms
4. Extracting entities for KG traversal

This is Step 1-2 of the retrieval pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from enum import Enum


class QueryIntent(Enum):
    """Detected query intent for retrieval optimization."""
    FACTUAL = "factual"  # Who, what, where, when
    COMPARISON = "comparison"  # Compare X vs Y
    TREND = "trend"  # Change, growth, decline over time
    TABLE_LOOKUP = "table_lookup"  # Explicit table/breakdown request
    DEFINITION = "definition"  # What is, define
    CALCULATION = "calculation"  # Sum, total, calculate
    GENERAL = "general"  # Default


@dataclass
class ProcessedQuery:
    """
    Query after processing and expansion.
    
    Contains all information needed for multi-path retrieval.
    """
    original: str  # Original user query
    normalized: str  # Cleaned, normalized query
    intent: QueryIntent  # Detected intent
    
    # Expanded queries for different search paths
    expanded_queries: List[str] = field(default_factory=list)
    
    # Extracted entities for KG traversal
    entities: List[str] = field(default_factory=list)
    
    # Detected financial/accounting terms
    domain_terms: List[str] = field(default_factory=list)
    
    # Time references (for temporal filtering)
    time_references: List[str] = field(default_factory=list)
    
    # If table lookup detected
    wants_table: bool = False
    
    # Metadata
    metadata: Dict = field(default_factory=dict)


class QueryProcessor:
    """
    Processes user queries for optimal retrieval.
    
    Financial/Legal Domain Adaptations:
    - Expands accounting terms (revenue ↔ sales ↔ net sales)
    - Recognizes SEC filing structure (Item 1, 1A, 7, etc.)
    - Detects financial entities (companies, metrics)
    - Handles numerical queries appropriately
    """
    
    # Financial term synonyms - bidirectional expansion
    FINANCIAL_SYNONYMS = {
        "revenue": ["sales", "net sales", "total revenue", "gross revenue", "turnover"],
        "profit": ["earnings", "net income", "net profit", "bottom line"],
        "expenses": ["costs", "expenditures", "operating expenses", "opex"],
        "assets": ["holdings", "resources", "capital"],
        "liabilities": ["debts", "obligations", "payables"],
        "equity": ["shareholders equity", "stockholders equity", "net worth"],
        "margin": ["profit margin", "gross margin", "operating margin"],
        "growth": ["increase", "rise", "expansion", "appreciation"],
        "decline": ["decrease", "reduction", "contraction", "depreciation"],
        "risk": ["risk factors", "risks", "uncertainties", "threats"],
        "ceo": ["chief executive officer", "chief executive"],
        "cfo": ["chief financial officer"],
        "auditor": ["independent auditor", "external auditor"],
        "filing": ["report", "submission", "disclosure"],
    }
    
    # SEC Item number patterns
    SEC_ITEMS = {
        "1": "business description",
        "1a": "risk factors",
        "1b": "unresolved staff comments",
        "2": "properties",
        "3": "legal proceedings",
        "4": "mine safety disclosures",
        "5": "market for common equity",
        "6": "selected financial data",
        "7": "management discussion analysis mda",
        "7a": "market risk disclosures",
        "8": "financial statements",
        "9": "changes disagreements accountants",
        "9a": "controls procedures",
        "9b": "other information",
        "10": "directors executive officers",
        "11": "executive compensation",
        "12": "security ownership",
        "13": "related transactions",
        "14": "principal accountant fees",
        "15": "exhibits financial schedules",
    }
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        QueryIntent.COMPARISON: [
            r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference\b",
            r"\bcompared to\b", r"\bbetter than\b", r"\bworse than\b"
        ],
        QueryIntent.TREND: [
            r"\btrend\b", r"\bchange\b", r"\bgrowth\b", r"\bdecline\b",
            r"\bover time\b", r"\byear over year\b", r"\byoy\b",
            r"\bincrease\b", r"\bdecrease\b"
        ],
        QueryIntent.TABLE_LOOKUP: [
            r"\btable\b", r"\bbreakdown\b", r"\bby segment\b", r"\bby region\b",
            r"\bdata\b", r"\bnumbers\b", r"\bfigures\b"
        ],
        QueryIntent.DEFINITION: [
            r"\bwhat is\b", r"\bdefine\b", r"\bdefinition\b", r"\bmeaning of\b",
            r"\bexplain\b"
        ],
        QueryIntent.CALCULATION: [
            r"\btotal\b", r"\bsum\b", r"\bcalculate\b", r"\bhow much\b",
            r"\bcount\b"
        ],
    }
    
    # Entity patterns
    COMPANY_PATTERN = re.compile(
        r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*'
        r'(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Corporation)\.?))\b'
    )
    
    MONEY_PATTERN = re.compile(
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?\b'
    )
    
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    
    def __init__(self):
        """Initialize query processor."""
        # Build reverse synonym map
        self._synonym_map = {}
        for key, synonyms in self.FINANCIAL_SYNONYMS.items():
            self._synonym_map[key] = synonyms
            for syn in synonyms:
                if syn not in self._synonym_map:
                    self._synonym_map[syn] = [key]
                else:
                    self._synonym_map[syn].append(key)
    
    def process(self, query: str) -> ProcessedQuery:
        """
        Process a user query for retrieval.
        
        This is the main entry point.
        
        Args:
            query: Raw user query
            
        Returns:
            ProcessedQuery with all processing results
        """
        # Normalize
        normalized = self._normalize(query)
        
        # Detect intent
        intent = self._detect_intent(normalized)
        
        # Expand query
        expanded = self._expand_query(normalized)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Find domain terms
        domain_terms = self._find_domain_terms(normalized)
        
        # Extract time references
        time_refs = self._extract_time_references(query)
        
        # Check for table request
        wants_table = intent == QueryIntent.TABLE_LOOKUP or self._wants_table(normalized)
        
        return ProcessedQuery(
            original=query,
            normalized=normalized,
            intent=intent,
            expanded_queries=expanded,
            entities=entities,
            domain_terms=domain_terms,
            time_references=time_refs,
            wants_table=wants_table
        )
    
    def _normalize(self, query: str) -> str:
        """
        Normalize query text.
        
        - Lowercase
        - Remove extra whitespace
        - Keep punctuation for now (might be meaningful)
        """
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent for retrieval optimization."""
        query_lower = query.lower()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent
        
        # Check for factual questions
        if re.match(r'^(who|what|where|when|which|how)\b', query_lower):
            return QueryIntent.FACTUAL
        
        return QueryIntent.GENERAL
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and domain terms.
        
        Creates multiple query variants for broader retrieval.
        """
        expanded = [query]  # Always include original
        
        words = query.lower().split()
        
        for word in words:
            # Check for financial synonyms
            if word in self._synonym_map:
                for synonym in self._synonym_map[word][:2]:  # Limit expansions
                    new_query = query.replace(word, synonym)
                    if new_query not in expanded:
                        expanded.append(new_query)
            
            # Check for SEC Item references
            item_match = re.match(r'item\s*(\d+[ab]?)', word)
            if item_match:
                item_num = item_match.group(1).lower()
                if item_num in self.SEC_ITEMS:
                    item_desc = self.SEC_ITEMS[item_num]
                    expanded.append(query + f" {item_desc}")
        
        # Add SEC Item expansion if relevant terms found
        for item_num, description in self.SEC_ITEMS.items():
            for desc_word in description.split():
                if desc_word in query.lower() and len(desc_word) > 3:
                    expanded.append(f"item {item_num} {query}")
                    break
        
        return expanded[:5]  # Limit to 5 variants
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities for knowledge graph traversal.
        
        Returns company names, metrics, and other entities.
        """
        entities = []
        
        # Company names
        companies = self.COMPANY_PATTERN.findall(query)
        entities.extend(companies)
        
        # Financial metrics mentioned
        metric_keywords = [
            "revenue", "profit", "income", "earnings", "margin",
            "assets", "liabilities", "equity", "cash flow", "debt"
        ]
        for metric in metric_keywords:
            if metric in query.lower():
                entities.append(metric)
        
        return list(set(entities))
    
    def _find_domain_terms(self, query: str) -> List[str]:
        """Find financial/legal domain terms in query."""
        terms = []
        
        # Check all known terms
        all_terms = set(self.FINANCIAL_SYNONYMS.keys())
        for synonyms in self.FINANCIAL_SYNONYMS.values():
            all_terms.update(synonyms)
        
        query_words = set(query.lower().split())
        terms = list(all_terms.intersection(query_words))
        
        return terms
    
    def _extract_time_references(self, query: str) -> List[str]:
        """Extract year and time references."""
        refs = []
        
        # Years
        years = self.YEAR_PATTERN.findall(query)
        refs.extend(years)
        
        # Relative time
        time_words = ["fiscal year", "quarter", "annual", "yearly", "monthly"]
        for tw in time_words:
            if tw in query.lower():
                refs.append(tw)
        
        return refs
    
    def _wants_table(self, query: str) -> bool:
        """Check if query is asking for tabular data."""
        table_indicators = [
            "table", "breakdown", "by segment", "by region",
            "by year", "by quarter", "figures", "data",
            "show me", "list of"
        ]
        
        return any(ind in query.lower() for ind in table_indicators)


def process_query(query: str) -> ProcessedQuery:
    """Convenience function to process a query."""
    processor = QueryProcessor()
    return processor.process(query)
