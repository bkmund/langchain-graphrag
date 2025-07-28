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

load_dotenv()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build a knowledge graph from a source text file.")
    parser.add_argument("input_file", type=str, help="Path to the source text file.")
    args = parser.parse_args()

    # ====================================
    # Setup for Knowledge Graph
    # ====================================
    print("--- Creating a Knowledge Graph ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    graph_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=True, # Allow extracting properties for nodes
        relationship_properties=True, # Allow extracting properties for relationships
    )

    print(f"Reading source text from: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as file:
        text = file.read()
    
    document = Document(page_content=text)


    print("Building Knowledge graph using LLMGraphTransformer...")
    graph_documents = graph_transformer.convert_to_graph_documents([document])
    
    extracted_graph = graph_documents[0]

    # ======================================
    # Saving the graph
    # ======================================
    nx_graph = NetworkxEntityGraph()

    for node in extracted_graph.nodes:
        # Create a dictionary of all attributes.
        all_properties = node.properties.copy() if node.properties else {}
        # Add the 'type' as just another attribute.
        all_properties['type'] = node.type
        # Pass all attributes in a single dictionary.
        nx_graph.add_node(node.id, **all_properties)

    for rel in extracted_graph.relationships:
        nx_graph.add_edge(rel.source.id, rel.target.id, relation=rel.type, **(rel.properties or {}))

    print_graph(nx_graph)

    # Save the graph to a JSON file
    graph_data = json_graph.node_link_data(nx_graph, edges="edges")
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_filename = f"{base_name}_graph.json"

    with open(output_filename, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"\nGraph saved as {output_filename}")

def print_graph(graph):
    """Helper function to print the graph's nodes and edges."""
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

if __name__ == "__main__":
    main()