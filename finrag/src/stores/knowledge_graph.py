"""
Neo4j Knowledge Graph - MANDATORY Component
============================================

This is a REQUIRED component of the RAG system.
The Knowledge Graph stores ONLY structured relationships, not raw text.

Purpose:
- Entity relationships (companies, metrics, dates)
- Section structure (document → section → chunk hierarchy)
- Table references (chunk → table linkage)
- Cross-document knowledge (entities appearing in multiple docs)

Every node/edge MUST reference a chunk_id for traceability.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum

# Neo4j driver import with fallback
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    print("[KnowledgeGraph] Neo4j driver not installed. Run: pip install neo4j")


class EntityType(Enum):
    """Types of entities extracted from documents."""
    COMPANY = "company"
    PERSON = "person"
    MONEY = "money"
    DATE = "date"
    PERCENTAGE = "percentage"
    METRIC = "metric"  # Revenue, profit, etc.
    LOCATION = "location"
    REGULATION = "regulation"
    RISK_FACTOR = "risk_factor"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Document structure
    CONTAINS = "CONTAINS"
    MENTIONS = "MENTIONS"
    REFERENCES_TABLE = "REFERENCES_TABLE"
    
    # Entity relationships
    REPORTS = "REPORTS"  # Company reports metric
    OPERATES_IN = "OPERATES_IN"  # Company operates in location
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    COMPETES_WITH = "COMPETES_WITH"
    RELATED_TO = "RELATED_TO"
    
    # Temporal
    INCREASED_FROM = "INCREASED_FROM"
    DECREASED_FROM = "DECREASED_FROM"
    CHANGED_TO = "CHANGED_TO"


@dataclass
class Entity:
    """
    An entity extracted from document text.
    
    Entities are nodes in the knowledge graph that represent
    real-world concepts (companies, metrics, dates).
    """
    entity_id: str
    name: str
    entity_type: EntityType
    normalized_name: str  # Standardized form for matching
    chunk_ids: List[str] = field(default_factory=list)  # Source chunks
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls, 
        name: str, 
        entity_type: EntityType,
        chunk_id: str
    ) -> "Entity":
        """Create entity with auto-generated ID."""
        normalized = cls._normalize_name(name)
        entity_id = f"{entity_type.value}_{normalized[:20]}".replace(" ", "_")
        
        return cls(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            normalized_name=normalized,
            chunk_ids=[chunk_id]
        )
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize entity name for matching."""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = name.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized


@dataclass
class Relation:
    """
    A relationship between two entities.
    
    All relations must reference the chunk_id where the
    relationship was observed.
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    chunk_id: str  # Source chunk where relation was found
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Neo4j-based Knowledge Graph for the RAG system.
    
    This is a MANDATORY component - the system is invalid without it.
    
    Node Types:
    - Document: Top-level document container
    - Section: Document sections (SEC Items, Parts)
    - Chunk: Atomic retrieval unit (references chunk_id)
    - Entity: Extracted entities (companies, metrics, etc.)
    - Table: Table references
    
    Edge Types:
    - CONTAINS: Hierarchical containment
    - MENTIONS: Chunk mentions entity
    - REFERENCES_TABLE: Chunk references table
    - Various entity relationships
    
    Usage:
        kg = KnowledgeGraph(uri, user, password)
        kg.add_document(doc)
        kg.add_chunks(chunks)
        kg.add_entities(entities)
        results = kg.traverse_from_entities(["Apple", "Revenue"])
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize connection to Neo4j.
        
        If Neo4j is not available, falls back to in-memory mode
        with limited functionality (for testing without Neo4j server).
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        self._driver = None
        self._connected = False
        
        # In-memory fallback storage
        self._memory_nodes: Dict[str, Dict] = {}
        self._memory_edges: List[Dict] = []
        
        # Try to connect
        self._connect()
    
    def _connect(self) -> bool:
        """Establish connection to Neo4j."""
        if not HAS_NEO4J:
            print("[KnowledgeGraph] Running in memory-only mode (Neo4j not installed)")
            return False
        
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connection
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            self._connected = True
            print(f"[KnowledgeGraph] Connected to Neo4j at {self.uri}")
            
            # Initialize schema
            self._init_schema()
            return True
            
        except Exception as e:
            print(f"[KnowledgeGraph] Neo4j connection failed: {e}")
            print("[KnowledgeGraph] Falling back to memory mode")
            self._connected = False
            return False
    
    def _init_schema(self) -> None:
        """Create indexes and constraints for optimal performance."""
        if not self._connected:
            return
        
        with self._driver.session(database=self.database) as session:
            # Unique constraints
            constraints = [
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT table_id IF NOT EXISTS FOR (t:Table) REQUIRE t.table_id IS UNIQUE",
                "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document node to the graph.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            filename: Source filename
            metadata: Additional properties
        """
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.title = $title,
                        d.filename = $filename,
                        d.metadata = $metadata
                """, doc_id=doc_id, title=title, filename=filename,
                    metadata=str(metadata or {}))
        else:
            self._memory_nodes[f"Document:{doc_id}"] = {
                "type": "Document",
                "doc_id": doc_id,
                "title": title,
                "filename": filename
            }
    
    def add_section(
        self,
        section_id: str,
        doc_id: str,
        title: str,
        item_number: Optional[str] = None,
        page_start: int = 0,
        page_end: int = 0
    ) -> None:
        """Add a section node linked to document."""
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("""
                    MERGE (s:Section {section_id: $section_id})
                    SET s.title = $title,
                        s.item_number = $item_number,
                        s.page_start = $page_start,
                        s.page_end = $page_end
                    WITH s
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:CONTAINS]->(s)
                """, section_id=section_id, doc_id=doc_id, title=title,
                    item_number=item_number, page_start=page_start, page_end=page_end)
        else:
            self._memory_nodes[f"Section:{section_id}"] = {
                "type": "Section",
                "section_id": section_id,
                "doc_id": doc_id,
                "title": title
            }
            self._memory_edges.append({
                "source": f"Document:{doc_id}",
                "target": f"Section:{section_id}",
                "type": "CONTAINS"
            })
    
    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        section_id: Optional[str],
        summary: str,
        page_start: int,
        page_end: int
    ) -> None:
        """
        Add a chunk node to the graph.
        
        IMPORTANT: Only stores chunk_id and summary, NOT full text.
        Full text is stored in Vector Store.
        """
        # Generate short summary if not provided
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.summary = $summary,
                        c.page_start = $page_start,
                        c.page_end = $page_end
                """, chunk_id=chunk_id, summary=summary,
                    page_start=page_start, page_end=page_end)
                
                # Link to section or document
                if section_id:
                    session.run("""
                        MATCH (s:Section {section_id: $section_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (s)-[:CONTAINS]->(c)
                    """, section_id=section_id, chunk_id=chunk_id)
                else:
                    session.run("""
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, doc_id=doc_id, chunk_id=chunk_id)
        else:
            self._memory_nodes[f"Chunk:{chunk_id}"] = {
                "type": "Chunk",
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "summary": summary
            }
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity node to the graph.
        
        Creates MENTIONS edges to all source chunks.
        """
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("""
                    MERGE (e:Entity {entity_id: $entity_id})
                    SET e.name = $name,
                        e.entity_type = $entity_type,
                        e.normalized_name = $normalized_name,
                        e.properties = $properties
                """, entity_id=entity.entity_id, name=entity.name,
                    entity_type=entity.entity_type.value,
                    normalized_name=entity.normalized_name,
                    properties=str(entity.properties))
                
                # Create MENTIONS edges
                for chunk_id in entity.chunk_ids:
                    session.run("""
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MATCH (e:Entity {entity_id: $entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, chunk_id=chunk_id, entity_id=entity.entity_id)
        else:
            self._memory_nodes[f"Entity:{entity.entity_id}"] = {
                "type": "Entity",
                "entity_id": entity.entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type.value
            }
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relationship between entities."""
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run(f"""
                    MATCH (s:Entity {{entity_id: $source_id}})
                    MATCH (t:Entity {{entity_id: $target_id}})
                    MERGE (s)-[r:{relation.relation_type.value}]->(t)
                    SET r.chunk_id = $chunk_id,
                        r.properties = $properties
                """, source_id=relation.source_id, target_id=relation.target_id,
                    chunk_id=relation.chunk_id, properties=str(relation.properties))
        else:
            self._memory_edges.append({
                "source": f"Entity:{relation.source_id}",
                "target": f"Entity:{relation.target_id}",
                "type": relation.relation_type.value,
                "chunk_id": relation.chunk_id
            })
    
    def add_table_reference(
        self,
        table_id: str,
        chunk_id: str,
        caption: Optional[str] = None
    ) -> None:
        """Link a table to its referencing chunk."""
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("""
                    MERGE (t:Table {table_id: $table_id})
                    SET t.caption = $caption
                    WITH t
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:REFERENCES_TABLE]->(t)
                """, table_id=table_id, chunk_id=chunk_id, caption=caption)
    
    def traverse_from_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2
    ) -> List[str]:
        """
        Traverse graph starting from entities, return related chunk_ids.
        
        This is used during retrieval when entities are detected in query.
        Finds chunks that mention these entities or related entities.
        
        Args:
            entity_names: List of entity names to start from
            max_hops: Maximum traversal depth
            
        Returns:
            List of chunk_ids related to the entities
        """
        if not self._connected:
            # Memory mode: simple matching
            chunk_ids = set()
            for node_key, node in self._memory_nodes.items():
                if node.get("type") == "Entity":
                    if any(name.lower() in node.get("name", "").lower() 
                           for name in entity_names):
                        # Find chunks mentioning this entity
                        for edge in self._memory_edges:
                            if edge.get("target") == node_key:
                                source = edge.get("source", "")
                                if source.startswith("Chunk:"):
                                    chunk_ids.add(source.replace("Chunk:", ""))
            return list(chunk_ids)
        
        # Neo4j traversal
        chunk_ids = []
        with self._driver.session(database=self.database) as session:
            # Normalize search names
            normalized_names = [Entity._normalize_name(n) for n in entity_names]
            
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.normalized_name IN $names
                CALL {
                    WITH e
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    RETURN c.chunk_id AS chunk_id
                    UNION
                    WITH e
                    MATCH (c:Chunk)-[:MENTIONS]->(:Entity)-[*1..2]-(e)
                    RETURN c.chunk_id AS chunk_id
                }
                RETURN DISTINCT chunk_id
                LIMIT 50
            """, names=normalized_names)
            
            chunk_ids = [record["chunk_id"] for record in result]
        
        return chunk_ids
    
    def get_section_context(self, chunk_id: str) -> Dict[str, Any]:
        """Get section and document context for a chunk."""
        if not self._connected:
            node = self._memory_nodes.get(f"Chunk:{chunk_id}", {})
            return {
                "chunk_id": chunk_id,
                "doc_id": node.get("doc_id"),
                "section_title": None
            }
        
        with self._driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk {chunk_id: $chunk_id})
                OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
                OPTIONAL MATCH (d:Document)-[:CONTAINS*]->(c)
                RETURN c.chunk_id as chunk_id,
                       s.title as section_title,
                       s.item_number as item_number,
                       d.doc_id as doc_id,
                       d.title as doc_title
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                return dict(record)
            return {"chunk_id": chunk_id}
    
    def get_chunk_entities(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a chunk."""
        if not self._connected:
            return []
        
        entities = []
        with self._driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
                RETURN e.entity_id as entity_id,
                       e.name as name,
                       e.entity_type as entity_type
            """, chunk_id=chunk_id)
            
            entities = [dict(record) for record in result]
        
        return entities
    
    def clear(self) -> None:
        """Clear all data from the graph."""
        if self._connected:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
        
        self._memory_nodes = {}
        self._memory_edges = []
        print("[KnowledgeGraph] Graph cleared")
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected
    
    @property
    def node_count(self) -> int:
        """Get total number of nodes."""
        if self._connected:
            with self._driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                return result.single()["count"]
        return len(self._memory_nodes)
