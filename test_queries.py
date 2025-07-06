import os
import json
import argparse

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain

from networkx.readwrite import json_graph

## ===============================
## Parser for Command-line
## ===============================

parser = argparse.ArgumentParser(
    description="Test queries against a saved knowledge graph JSON file.")
parser.add_argument("graph_file", type=str, help="Path to the saved graph JSON file.")
args = parser.parse_args()

## ===============================
## Setup 
## ===============================

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

## ===============================
## Loading graph from json file
## ===============================

print(f"Loading graph from {args.graph_file}...")

with open(args.graph_file, 'r', encoding='utf-8') as f:
    graph_data = json.load(f)

nx_graph = json_graph.node_link_graph(graph_data, directed=True, multigraph=True, edges="edges")

graph_wrapper = NetworkxEntityGraph()
graph_wrapper._graph = nx_graph

print("Graph loaded successfully.")
print("-" * 20)

chain = GraphQAChain.from_llm(llm, graph=graph_wrapper, verbose=True)

# Asking queries to the LLM
with open('queries.txt', 'r') as f:
    queries = [line.strip() for line in f if line.strip()]

for i, query in enumerate(queries):
    print(f"\n--- Running Query #{i+1}: '{query}' ---")
    
    response = chain.invoke(query)
    
    print("\n--- Answer ---")
    print(response["result"])
    print("="*20)