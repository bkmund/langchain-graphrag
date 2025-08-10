import os
import json
import argparse
import networkx as nx

from dotenv import load_dotenv
from llm_transformer_build_graph import *

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
# from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from networkx.readwrite import json_graph

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHUNK_SAVE_FILE = "chunks.json"

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

    # ======================================
    # Chunking document
    # ======================================

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_documents = text_splitter.create_documents([text])
    print(f"Split document into {len(chunked_documents)} chunks.")

    chunks_for_retrieval = [{"chunk_id": i, "text": doc.page_content} for i, doc in enumerate(chunked_documents)]
    with open(CHUNK_SAVE_FILE, "w") as f:
        json.dump(chunks_for_retrieval, f, indent=2)
    print(f"\nSaved text chunks to {CHUNK_SAVE_FILE}")

    #document = Document(page_content=text)
    #print("Building Knowledge graph using LLMGraphTransformer...")
    #graph_documents = graph_transformer.convert_to_graph_documents([document])
    
    #extracted_graph = graph_documents[0]

    # ======================================
    # Saving the graph
    # ======================================
    nx_graph = nx.DiGraph()

    for i, doc in enumerate(chunked_documents):
        print(f"Processing chunk {i}...")
        graph_document = graph_transformer.convert_to_graph_documents([doc])[0]

        for node in graph_document.nodes:
            # Create a dictionary of all attributes.
            all_properties = node.properties.copy() if node.properties else {}
            # Add the 'type' as just another attribute.
            all_properties['type'] = node.type
            # Add chunk ID as a property
            all_properties['source_chunk_id'] = i
            # Pass all attributes in a single dictionary.
            nx_graph.add_node(node.id, **all_properties)

        for rel in graph_document.relationships:
            rel_properties = rel.properties.copy() if rel.properties else {}
            # Add chunk ID as property
            rel_properties['source_chunk_id'] = i
            nx_graph.add_edge(rel.source.id, rel.target.id, relation=rel.type, **(rel.properties or {}))

    print_graph(nx_graph)

    # Save the graph to a JSON file
    graph_data = json_graph.node_link_data(nx_graph, edges="edges")
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_filename = f"{base_name}_graph_chunked.json"

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