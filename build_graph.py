import os
import json
import argparse

from dotenv import load_dotenv
from llm_transformer_build_graph import *

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from networkx.readwrite import json_graph

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

## ===================================
## Parse for Command-line
## ===================================

parser = argparse.ArgumentParser(
    description="Build a knowledge graph from a source text file.")
parser.add_argument("input_file", type=str, help="Path to the source text file.")
args = parser.parse_args()

## ====================================
## Setup for Knowledge Graph
## ====================================



print("--- Creating a Knowledge Graph ---")
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

print(f"Reading source text from: {args.input_file}")
with open(args.input_file, "r", encoding="utf-8") as file:
    text = file.read()
documents = [Document(page_content=text)]

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
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

graph_data = json_graph.node_link_data(nx_graph, edges="edges")
base_name = os.path.splitext(os.path.basename(args.input_file))[0]
output_filename = f"{base_name}_graph.json"

with open(output_filename, "w") as f:
    json.dump(graph_data, f)

print(f"\nGraph saved as {output_filename}")