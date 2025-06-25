import os
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

with open('knowledge_graph_marie.pkl', 'rb') as f:
    graph_wrapper = pickle.load(f)

nx_graph = graph_wrapper._graph
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