import os
import json
import argparse

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

from networkx.readwrite import json_graph

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

## ===============================
## Parser for Command-line
## ===============================

with open('chunks.json', 'r') as f:
    CHUNK_DATABASE = {chunk['chunk_id']: chunk['text'] for chunk in json.load(f)}

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

## ===============================
## RAG chain
## ===============================

def get_entities(question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting entities from a user question. Only extract the entities, no other text."),
        ("human", "Extract the key entities from the following question: {question}")
    ])
    entity_extraction_chain = prompt | llm | StrOutputParser()
    return entity_extraction_chain.invoke({"question": question})

def format_graph_context(graph, entities_string):
    entities = [e.strip() for e in entities_string.split(',')]
    context = ""
    for entity in entities:
        if not graph.has_node(entity):
            continue
            
        node_data = graph.nodes[entity]
        context += f"Node '{entity}' has properties: {json.dumps(node_data, indent=2)}\n"
        
        for source, target, data in graph.edges(entity, data=True):
            relation_props = {k: v for k, v in data.items() if k != 'relation'}
            context += f"OUTGOING: ({source}) -[{data.get('relation', 'N/A')}]-> ({target})"
            if relation_props:
                context += f" with properties: {json.dumps(relation_props)}\n"
            else:
                context += "\n"

        for source, target, data in graph.in_edges(entity, data=True):
            relation_props = {k: v for k, v in data.items() if k != 'relation'}
            context += f"INCOMING: ({source}) -[{data.get('relation', 'N/A')}]-> ({target})"
            if relation_props:
                context += f" with properties: {json.dumps(relation_props)}\n"
            else:
                context += "\n"
                
    return context

def retrieve_chunks_from_ids(chunk_ids: list) -> str:
    """Retrieves the text of specified chunks from the global chunk database."""
    retrieved_texts = [CHUNK_DATABASE.get(cid, "") for cid in chunk_ids]
    return "\n\n---\n\n".join(filter(None, retrieved_texts))

def get_relevant_chunk_ids(inputs: dict) -> list:
    """
    Uses an LLM to look at the graph context and question,
    then decides which source_chunk_id to retrieve.
    """
    chunk_id_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a routing assistant. Based on the user's question and the provided knowledge graph summary,
        your task is to identify the most relevant 'source_chunk_id' to retrieve for a detailed answer.
        The graph summary contains properties that include 'source_chunk_id'.
        Respond ONLY with a comma-separated list of integer chunk IDs. If no specific chunk seems relevant, respond with an empty string.
        """),
        ("human", "Question: {question}\n\nGraph Summary:\n{graph_context}")
    ])

    # A simple chain to parse the LLM's output into a list of integers
    id_chain = chunk_id_prompt | llm | StrOutputParser()
    id_string = id_chain.invoke({
        "question": inputs["question"],
        "graph_context": inputs["graph_context"]
    })
    
    if not id_string:
        return []
    
    # Clean up and return a list of integers
    return [int(cid.strip()) for cid in id_string.split(',') if cid.strip().isdigit()]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a highly intelligent question-answering assistant that operates on a knowledge graph.
    Your task is to answer the user's question based *only* on the context provided.

    The context you receive is a structured summary of this graph. It contains:
    - Nodes (e.g., 'Marie Curie') and their properties (e.g., 'birth_year': 1867).
    - Relationships connecting them (e.g., OUTGOING: (Marie Curie) -[SPOUSE]-> (Pierre Curie)).

    **Your Rules:**
    1.  **Interpret Properties:** You must carefully analyze the properties of the nodes to answer questions. For example, if a node has a property `"birth_year": 1867`, you can definitively answer when that person was born.
    2.  **Follow Relationships:** Use the relationships to understand how entities are connected.
    3.  **Be Precise:** Your answer must be directly and provably supported by the information in the context.
    4.  **Do Not Infer:** If the context does not contain the information to answer the question (e.g., the context mentions an award but not the year it was won), you MUST respond with the exact phrase: "I don't know." Do not use any of your outside knowledge.
    """),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            graph_context=lambda x: format_graph_context(nx_graph, get_entities(x["question"]))
        )
        | RunnablePassthrough.assign(
            chunk_ids=get_relevant_chunk_ids
        )
        | RunnablePassthrough.assign(
            retrieved_chunks=lambda x: retrieve_chunks_from_ids(x["chunk_ids"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

## ===================================
## Asking queries to LLM
## ===================================

with open('queries.txt', 'r') as f:
    queries = [line.strip() for line in f if line.strip()]

for i, query in enumerate(queries):
    print(f"\n--- Running Query #{i+1}: '{query}' ---")
    debug_context = format_graph_context(nx_graph, get_entities(query))
    print(f"DEBUG CONTEXT:\n{debug_context}")
    response = rag_chain.invoke(query)
    
    print("\n--- Answer ---")
    print(response)
    print("="*20)