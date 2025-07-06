import os
import json
import pickle
import networkx as nx

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("--- Creating a Knowledge Graph ---")
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

with open("source_text.txt", "r", encoding="utf-8") as file:
    text = file.read()
documents = [Document(page_content=text)]

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization", "Prize"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE", "AWARDED"],
    node_properties=["Birth_Year"],
    relationship_properties=["Start_Year", "Year_Awarded", "Award_Field"]
    )
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)

print("--- Raw Extracted Graph Document ---")
print(graph_documents_filtered[0])
print("-" * 20)

source_graph = graph_documents_filtered[0]
graph_wrapper = NetworkxEntityGraph()
nx_graph = graph_wrapper._graph

for node in source_graph.nodes:
    nx_graph.add_node(node.id, type=node.type, **node.properties)

for rel in source_graph.relationships:
    nx_graph.add_edge(rel.source.id, rel.target.id, relation=rel.type, **rel.properties)
    
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