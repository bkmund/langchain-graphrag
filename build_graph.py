import os
import json
import pickle

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field

## ==================================
## Defining a schema for LLM output
## ==================================

class Node(BaseModel):
    """Represents a single entity in the graph."""
    id: str = Field(description="The unique name or identifier of the entity.")
    type: str = Field(description="The type of the entity (e.g., Person, Country, Organization, Prize).")
    properties: Optional[dict] = Field(
        default_factory=dict,
        description="A dictionary of properties for the node. Properties must be explicitly mentioned in the source text."
    )

class Relationship(BaseModel):
    """Represents a relationship between two entities."""
    source: str = Field(description="The ID of the source node.")
    target: str = Field(description="The ID of the target node.")
    type: str = Field(description="The type of the relationship (e.g., SPOUSE, AWARDED).")
    properties: Optional[dict] = Field(
        default_factory=dict,
        description="A dictionary of properties for the relationship. Properties must be explicitly mentioned in the source text."
    )

class KnowledgeGraph(BaseModel):
    """A knowledge graph extracted from a text, adhering to a strict schema."""
    nodes: List[Node] = Field(description="A list of all unique entities from the text.")
    relationships: List[Relationship] = Field(
        description="A list of all relationships between entities from the text."
    )

## ====================================
## Setup for Knowledge Graph
## ====================================


print("--- Creating a Knowledge Graph ---")
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

with open("source_text.txt", "r", encoding="utf-8") as file:
    text = file.read()
documents = [Document(page_content=text)]

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert data extraction engine. Your sole task is to extract a knowledge graph from the provided text.
    You must follow these rules without exception:
    
    1.  **ONLY USE THE PROVIDED TEXT:** Extract nodes, relationships, and properties mentioned *exclusively* in the source text.
    2.  **ADHERE TO THE SCHEMA:** The output MUST be a JSON object that strictly follows the provided `KnowledgeGraph` schema.
    3.  **DO NOT HALLUCINATE:** If a property (like 'birth_year' or 'year_awarded') is not explicitly stated in the text for a given entity or relationship, you MUST NOT include it in the properties dictionary.
    4.  **CRITICAL RULE:** Do not invent, infer, or use any of your pre-trained knowledge. If the text does not contain the information, the output should not contain it.
    """),
    ("human", "Extract a knowledge graph from the following text:\n\n{text}")
])

extraction_chain = extraction_prompt | llm.with_structured_output(
    KnowledgeGraph, method = "function_calling")
print("Bulding Knowledge graph...")
extracted_graph = extraction_chain.invoke({"text": text})

## ======================================
## Saving the graph
## ======================================

graph_wrapper = NetworkxEntityGraph()
nx_graph = graph_wrapper._graph

for node in extracted_graph.nodes:
    nx_graph.add_node(node.id, type=node.type, **(node.properties or {}))

for rel in extracted_graph.relationships:
    nx_graph.add_edge(rel.source, rel.target, relation=rel.type, **(rel.properties or {}))
    
def print_graph(graph):
    print("\n--- Nodes ---")
    for node, properties in graph.nodes(data=True):
        print(f"- {node:<20} | Properties: {json.dumps(properties)}")
    print("\n--- Edges ---")
    for source, target, properties in graph.edges(data=True):
        relation = properties.get('relation', 'N/A')
        other_props = {k: v for k, v in properties.items() if k != 'relation'}
        print(f"- ({source}) -[{relation}]-> ({target})")
        if other_props:
            print(f"    Properties: {json.dumps(other_props)}")

print_graph(nx_graph)

with open("knowledge_graph_marie.pkl", "wb") as f:
    pickle.dump(graph_wrapper, f)

print("Graph saved as knowledge_graph_marie.pkl")