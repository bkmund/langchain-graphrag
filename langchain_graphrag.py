import os
import pickle
import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

with open("windows_text.md", "r", encoding="utf-8") as file:
    text = file.read()

documents = [Document(page_content=text)]
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
    node_properties=True,
    relationship_properties=True
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
    
    #triple = KnowledgeTriple(subject = rel.source.id, object_=rel.target.id, predicate=rel.type)
    #graph.add_triple(triple)

print ("---Graph---")
print("\n--- Nodes ---")
for node in graph_wrapper._graph.nodes(data=True):
    print(node)

print("\n--- Edges ---")
for edge in graph_wrapper._graph.edges(data=True):
    print(edge)

with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump(graph_wrapper, f)

print("Graph saved as knowledge_graph.pkl")

