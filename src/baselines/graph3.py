import sys
import os
import asyncio
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import concurrent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
sys.path.append('.')

from src.langchain_util import init_langchain_model, num_tokens_by_tiktoken
from src.processing import mean_pooling_embedding_with_normalization
from src.elastic_search_tool import search_with_score

from ircot import (
    DocumentRetriever, BM25Retriever, DPRRetriever, SentenceTransformersRetriever,
    Colbertv2Retriever, parse_prompt, merge_elements_with_same_first_line
)
import pdb

import numpy as np
import torch
import faiss
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from openai import OpenAI, AsyncOpenAI
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


es = Elasticsearch(
    "http://10.7.0.4:9200",  
    basic_auth=("elastic", "0qSJ7FdnqR23y8bMoyra")  
)

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'

class NodeType(str, Enum):
    retrievalandreasoning = "retrievalandreasoning"
    reasoning = "reasoning"

class ExecutionNode(BaseModel):
    id: int
    instruction: str
    node_type: NodeType
    # retrieval_query: Optional[str] = None
    upstream_node_ids: List[int] = Field(default_factory=list)

class ExecutionGraph(BaseModel):
    nodes: List[ExecutionNode]
    # root_node_id: int
    final_node_id: int

class GraphResponse(BaseModel):
    graph: ExecutionGraph
    explanation: str

async def construct_execution_graph(question: str, num_trials: int = 10) -> Tuple[GraphResponse, List[GraphResponse]]:
    client = AsyncOpenAI()
    system_message = """
    You are an AI assistant specialized in creating execution graphs for complex, multi-hop retrieval and reasoning tasks. Your role is to break down questions into a series of interconnected steps, represented as nodes in a graph.

    Consider the following when creating the graph:
    1. Break the task into logical, manageable steps
    2. Ensure that retrieval steps gather all necessary information
    3. Include reasoning steps to process and analyze the retrieved information
    4. Allow for parallel execution of independent steps where possible
    5. Ensure the final node synthesizes all necessary information to answer the original question
    """

    user_message = f"Create an execution graph for the following question: {question}"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


    all_graph_responses = []

    for _ in range(num_trials):
        try:
            completion = await client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=GraphResponse,
            )
            graph_response = completion.choices[0].message.parsed
            all_graph_responses.append(graph_response)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    return all_graph_responses

async def generate_instruction(node: ExecutionNode, client: Any) -> str:
    """
    generate instruction when just fix the graph structure
    """
    prompt = f"Rewrite the following instruction based on its context and upstream nodes:\n\n"
    prompt += f"Original instruction: {node.instruction}\n"
    if node.upstream_node_ids:
        prompt += f"Upstream node IDs: {node.upstream_node_ids}\n"
    prompt += "Rewrite Output:"

    messages = ChatPromptTemplate.from_messages([
        HumanMessage(prompt)
    ]).format_prompt()

    try:
        instruction_response = await client.ainvoke(messages.to_messages())
        return instruction_response.content.strip()
    except Exception as e:
        print(f"Error generating instruction: {e}")
        return node.instruction  # 如果生成失败，返回原始指令

async def retrieve_and_reason_step(query: str, instruction: str, corpus: Dict[str, Any], top_k: int, retriever: DocumentRetriever, dataset: str, client: Any, few_shot: List[Dict[str, Any]], upstream_results: List[Tuple[str, str]]) -> Tuple[str, List[str], List[float]]:
    # TODO think about how we can use the global query into line 101 
    # TODO Instruction should have explicit placeholder for the upstream results, please print the instruction
    # Retrieval query writing step
    prompt_user = f'Instruction: {instruction}\n'
    prompt_user += "To answer this questin, we executed the folloing upstream task first:\n"
    for node, result in upstream_results:
        prompt_user += f"This is the upstream task {node} And the result is {result}\n"
    messages = ChatPromptTemplate.from_messages([
        SystemMessage("You help write a retrieval query to gather relevant information for the reasoning task.  Return the query directly. Do not say 'Sure, here\'s a query'"),
        HumanMessage(prompt_user),
    ]).format_prompt()

    retrieval_query_current_step = (await client.ainvoke(messages.to_messages())).content
    # Retrieval step
    doc_ids, scores = await asyncio.to_thread(retriever.rank_docs, retrieval_query_current_step, top_k=top_k)
    
    if dataset in ['hotpotqa']:
        retrieved_passages = []
        for doc_id in doc_ids:
            key = list(corpus.keys())[doc_id]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    elif dataset in ['musique', '2wikimultihopqa']:
        retrieved_passages = [corpus[doc_id]['title'] + '\n' + corpus[doc_id]['text'] for doc_id in doc_ids]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

    # Reasoning step based on 
    prompt_user = f'Instruction: {instruction}\n'
    prompt_user += "Relevant information:\n"

    for passage in retrieved_passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += "Reasoned Output:"

    messages = ChatPromptTemplate.from_messages([
        HumanMessage(prompt_user)
    ]).format_prompt()

    try:
        chat_completion = await client.ainvoke(messages.to_messages())
        return chat_completion.content, retrieved_passages, scores
    except Exception as e:
        print(f"Error in retrieve and reason step: {e}")
        return '', retrieved_passages, scores

async def reason_step(instruction: str, client: Any, upstream_results: List[Tuple[str, str]]) -> str:
    prompt_user = f'Instruction: {instruction}\n\n To answer this questin, we executed the folloing upstream task first:\n'
    for node, result in upstream_results:
        prompt_user += f"This is the upstream task {node} And the result is {result}\n"
    prompt_user += "Reasoned Output:"

    messages = ChatPromptTemplate.from_messages([
        HumanMessage(prompt_user)
    ]).format_prompt()

    try:
        chat_completion = await client.ainvoke(messages.to_messages())
        return chat_completion.content
    except Exception as e:
        print(f"Error in reason step: {e}")
        return ''

async def execute_fixed_graph(graph_response,sample: Dict[str, Any], query, corpus, retriever, client, args, fixed_instruction=True):
    retrieved_passages_dict = {}
    thoughts = []
    node_outputs = {}

    for node in graph_response.graph.nodes:
        upstream_results = [(up_node.id, node_outputs[up_node.id]) for up_node in graph_response.graph.nodes if up_node.id in node.upstream_node_ids]

        if node.node_type == NodeType.retrievalandreasoning:
            instruction = node.instruction if fixed_instruction else await generate_instruction(node, client)
            result, passages, scores = await retrieve_and_reason_step(
                query=query,
                instruction=instruction,
                corpus=corpus,
                top_k=args.top_k,
                retriever=retriever,
                dataset=args.dataset,
                client=client,
                few_shot=few_shot_samples,
                upstream_results=upstream_results
            )
            for passage, score in zip(passages, scores):
                if passage in retrieved_passages_dict:
                    retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                else:
                    retrieved_passages_dict[passage] = score
        elif node.node_type == NodeType.reasoning:
            instruction = node.instruction if fixed_instruction else await generate_instruction(node, client)
            result = await reason_step(
                instruction=instruction,
                client=client,
                upstream_results=upstream_results
            )
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

        thoughts.append(result)
        node_outputs[node.id] = result

    sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
    retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
    
    if args.dataset in ['hotpotqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    elif args.dataset in ['musique']:
        gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
        gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
        retrieved_items = list(retrieved_passages)
    elif args.dataset in ['2wikimultihopqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    
    recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}

    avg_recall = sum(recall.values()) / len(recall)

    return {
        "node_id": node.id,
        "instruction": instruction,
        # "nodes": [],
        "output": result,
        "node_type": node.node_type.value,
        "retrieved_passages": list(retrieved_passages),
        "recall": avg_recall,
        "thoughts": thoughts,
        "upstream_results": upstream_results
    }

# async def process_sample(idx: int, sample: Dict[str, Any], args: argparse.Namespace, corpus: Dict[str, Any], retriever: DocumentRetriever, client: Any, processed_ids: set) -> Optional[Tuple[int, Dict[int, float], List[str], List[str], int]]:
#     if args.dataset in ['hotpotqa', '2wikimultihopqa']:
#         sample_id = sample['_id']
#     elif args.dataset in ['musique']:
#         sample_id = sample['id']
#     else:
#         raise NotImplementedError(f'Dataset {args.dataset} not implemented')

#     if sample_id in processed_ids:
#         return None  # Skip already processed samples

#     query = sample['question']
    
#     print(f"Processing Sample {idx}: Question - {query}")
    
#     all_graph_responses = await construct_execution_graph(query, num_trials=10)

#     if not all_graph_responses:
#         print(f"Failed to construct execution graph for sample {idx}")
#         return None
    
#     all_results = []
#     best_recall_score = -1
#     best_result = None
#     best_graph = None
#     best_trial = -1
    
#     for trial, graph_response in enumerate(all_graph_responses):
#         retrieved_passages_dict = {}
#         thoughts = []
#         node_outputs = {}

#         async def execute_node(node: ExecutionNode) -> str:
#             if node.id in node_outputs:
#                 return node_outputs[node.id]

#             upstream_results = []
#             for up_id in node.upstream_node_ids:
#                 up_node = next(n for n in graph_response.graph.nodes if n.id == up_id)
#                 up_result = await execute_node(up_node)
#                 upstream_results.append((up_node.id, up_result))

#             if node.node_type == NodeType.retrievalandreasoning:
#                 result, passages, scores = await retrieve_and_reason_step(
#                     query=query,
#                     instruction=node.instruction,
#                     corpus=corpus,
#                     top_k=args.top_k,
#                     retriever=retriever,
#                     dataset=args.dataset,
#                     client=client,
#                     few_shot=few_shot_samples,
#                     upstream_results=upstream_results
#                 )
#                 for passage, score in zip(passages, scores):
#                     if passage in retrieved_passages_dict:
#                         retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
#                     else:
#                         retrieved_passages_dict[passage] = score
#             elif node.node_type == NodeType.reasoning:
#                 result = await reason_step(
#                     instruction=node.instruction,
#                     client=client,
#                     upstream_results=upstream_results
#                 )
#             else:
#                 raise ValueError(f"Unknown node type: {node.node_type}")

#             thoughts.append(result)
#             node_outputs[node.id] = result
#             print(f"Node {node.id} output: {result}")
#             return result

#         # try except block to catch any errors that occur during execution
#         # check if the graph is dag
#         # final_node = next(n for n in graph_response.graph.nodes if n.id == graph_response.graph.final_node_id)
#         try:
#             final_node = next(n for n in graph_response.graph.nodes if n.id == graph_response.graph.final_node_id)
#         except StopIteration:
#             print(f"Error: Unable to find final_node for sample {idx}")
#             print(f"Graph content: {graph_response.graph}")
#             return None
#         # final_output = await execute_node(final_node)
#         await execute_node(final_node)
        
#         sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
#         retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])

#         if args.dataset in ['hotpotqa']:
#             gold_passages = [item for item in sample['supporting_facts']]
#             gold_items = set([item[0] for item in gold_passages])
#             retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
#         elif args.dataset in ['musique']:
#             gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
#             gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
#             retrieved_items = list(retrieved_passages)
#         elif args.dataset in ['2wikimultihopqa']:
#             gold_passages = [item for item in sample['supporting_facts']]
#             gold_items = set([item[0] for item in gold_passages])
#             retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
#         else:
#             raise NotImplementedError(f'Dataset {args.dataset} not implemented')

#         recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}

#         avg_recall = sum(recall.values()) / len(recall)

#         all_results.append({
#             "question": query,
#             "graph": graph_response.dict(),
#             "recall": avg_recall,
#             "retrieved_passages": list(retrieved_passages),
#             "thoughts": thoughts,
#             "is_best": False
#         })

#         if avg_recall > best_recall_score:
#             best_recall_score = avg_recall
#             best_result = (idx, recall, list(retrieved_passages), thoughts, len(thoughts))
#             best_graph = graph_response
#             best_trial = trial

#     if best_trial >= 0:
#         all_results[best_trial]["is_best"] = True
    
#     # additional_attempts_1 = []
#     # for i in range(10):
#     #     attempt_data = await execute_fixed_graph(best_graph,sample, query, corpus, retriever, client, args, fixed_instruction=False)
#     #     additional_attempts_1.append(attempt_data)
#     # with open(f'result/llm_20/fixed_graph_attempts_{idx}.json', 'w') as f:
#     #     json.dump(additional_attempts_1, f, indent=4)

#     # # Use best graph to generate 10 more attempts and save results
#     additional_attempts_2 = []
#     for i in range(10):
#         attempt_data = {
#             "attempt": i + 1,
#             "question": query,
#             "nodes": [],
#             "retrieved_passages": [],
#             "recall": None
#         }
#         retrieved_passages_dict = {}
#         thoughts = []
#         node_outputs = {}

#         for node in best_graph.graph.nodes:
#             upstream_results = [(up_node.id, node_outputs[up_node.id]) for up_node in best_graph.graph.nodes if up_node.id in node.upstream_node_ids]
#             if node.node_type == NodeType.retrievalandreasoning:
#                 result, passages, scores = await retrieve_and_reason_step(
#                     query=query,
#                     instruction=node.instruction,
#                     corpus=corpus,
#                     top_k=args.top_k,
#                     retriever=retriever,
#                     dataset=args.dataset,
#                     client=client,
#                     few_shot=few_shot_samples,
#                     upstream_results=upstream_results
#                 )
#                 for passage, score in zip(passages, scores):
#                     if passage in retrieved_passages_dict:
#                         retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
#                     else:
#                         retrieved_passages_dict[passage] = score
#             elif node.node_type == NodeType.reasoning:
#                 result = await reason_step(
#                     instruction=node.instruction,
#                     client=client,
#                     upstream_results=upstream_results
#                 )
#             else:
#                 raise ValueError(f"Unknown node type: {node.node_type}")

#             thoughts.append(result)
#             node_outputs[node.id] = result
#             attempt_data["nodes"].append({
#                 "node_id": node.id,
#                 "instruction": node.instruction,
#                 "output": result,
#                 "node_type": node.node_type.value,
#                 "upstream_node_ids": node.upstream_node_ids
#             })

#         sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
#         retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
#         attempt_data["retrieved_passages"] = list(retrieved_passages)

#         if args.dataset in ['hotpotqa']:
#             gold_passages = [item for item in sample['supporting_facts']]
#             gold_items = set([item[0] for item in gold_passages])
#             retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
#         elif args.dataset in ['musique']:
#             gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
#             gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
#             retrieved_items = list(retrieved_passages)
#         elif args.dataset in ['2wikimultihopqa']:
#             gold_passages = [item for item in sample['supporting_facts']]
#             gold_items = set([item[0] for item in gold_passages])
#             retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
#         else:
#             raise NotImplementedError(f'Dataset {args.dataset} not implemented')

#         recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}
#         avg_recall = sum(recall.values()) / len(recall)
#         attempt_data["recall"] = avg_recall

#         additional_attempts_2.append(attempt_data)

#     # Save best graph and additional attempts
#     with open(f'result/llm_20/best_graph_additional_{idx}.json', 'w') as f:
#         json.dump(additional_attempts_2, f, indent=4)

#     with open(f'result/llm_20/graph_attempts_all_{idx}.json', 'w') as f:
#         json.dump(all_results, f, indent=4)

#     print(f"The best result for Sample {idx} was from trial {best_trial}")
    
#     return best_result

async def process_sample(idx: int, sample: Dict[str, Any], args: argparse.Namespace, corpus: Dict[str, Any], retriever: DocumentRetriever, client: Any, processed_ids: set):
    if args.dataset in ['hotpotqa', '2wikimultihopqa']:
        sample_id = sample['_id']
    elif args.dataset in ['musique']:
        sample_id = sample['id']
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    
    if sample_id in processed_ids:
        return None
    
    query = sample['question']
    print(f"Processing Sample {idx}: Question - {query}")
    
    all_graph_responses = await construct_execution_graph(query, num_trials=10)

    if not all_graph_responses:
        print(f"Failed to construct execution graph for sample {idx}")
        return None
    
    all_results = []
    best_recall_score = -1
    best_result = None
    best_graph = None
    best_trial = -1
    
    for trial, graph_response in enumerate(all_graph_responses):
        retrieved_passages_dict = {}
        thoughts = []
        node_outputs = {}
    
        async def execute_node(node: ExecutionNode) -> str:
            if node.id in node_outputs:
                return node_outputs[node.id]
            
            upstream_results = []
            for up_id in node.upstream_node_ids:
                up_node = next(n for n in graph_response.graph.nodes if n.id == up_id)
                up_result = await execute_node(up_node)
                upstream_results.append((up_node.id, up_result))
                
            if node.node_type == NodeType.retrievalandreasoning:
                result, passages, scpres = await retrieve_and_reason_step(
                    query=query,
                    instruction=node.instruction,
                    corpus=corpus,
                    top_k=args.top_k,
                    retriever=retriever,
                    dataset=args.dataset,
                    client=client,
                    few_shot=few_shot_samples,
                    upstream_results=upstream_results
                )
                scores = None  
                for passage, score in zip(passages, scores):
                    if passage in retrieved_passages_dict:
                        retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                    else:
                        retrieved_passages_dict[passage] = score
            elif node.node_type == NodeType.reasoning:
                result = await reason_step(
                    instruction=node.instruction,
                    client=client,
                    upstream_results=upstream_results
                )
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
            
            thoughts.append(result)
            node_outputs[node.id] = result
            print(f"Node {node.id} output: {result}")
            return result
        
        try:
            final_node = next(n for n in graph_response.graph.nodes if n.id == graph_response.graph.final_node_id)
        except StopIteration:
            print(f"Error: Unable to find final_node for sample {idx}")
            print(f"Graph content: {graph_response.graph}")
            return None
        
        await execute_node(final_node)
        
        sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
        
        if args.dataset in ['hotpotqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif args.dataset in ['musique']:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
            retrieved_items = list(retrieved_passages)
        elif args.dataset in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        
        recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}
        
        avg_recall = sum(recall.values()) / len(recall)
        
        all_results.append({
            "question": query,
            "graph": graph_response.dict(),
            "recall": avg_recall,
            "retrieved_passages": list(retrieved_passages),
            "thoughts": thoughts,
            "is_best": False
        })
        
        if avg_recall > best_recall_score:
            best_recall_score = avg_recall
            best_result = (idx, recall, list(retrieved_passages), thoughts, len(thoughts))
            best_graph = graph_response
            best_trial = trial
            
    if best_trial >= 0:
        all_results[best_trial]["is_best"] = True
        
    with open(f'result/llm_20/graph_attempts_all_{idx}.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    # find out the best, fifth and lowest recall
    recalls = [(result["recall"], result) for result in all_results]
    recalls.sort(key=lambda x: x[0], reverse=True)
    
    best_result = recalls[0][1]
    fifth_result = recalls[4][1]
    lowest_result = recalls[-1][1]
    
    # find out the graph of the best, fifth and lowest recall
    best_graph = next((result["graph"] for result in all_results if result["is_best"]), None)
    fifth_graph = next((result["graph"] for result in all_results if result[1] == fifth_result), None)
    lowest_graph = next((result["graph"] for result in all_results if result[1] == lowest_result), None)
    
    # fix the instruction and structure of the best, do 10 more retrievals
    additional_attempts_best = []
    for i in range(10):
        attempt_data = {
            "attempt": i + 1,
            "question": query,
            "nodes": [],
            "retrieved_passages": [],
            "recall": None
        }
        retrieved_passages_dict = {}
        thoughts = []
        node_outputs = {}
        
        for node in best_graph.graph.nodes:
            upstream_results = [(up_node.id, node_outputs[up_node.id]) for up_node in best_graph.graph.nodes if up_node.id in node.upstream_node_ids]
            if node.node_type == NodeType.retrievalandreasoning:
                result, passages, scores = await retrieve_and_reason_step(
                    query=query,
                    instruction=node.instruction,
                    corpus=corpus,
                    top_k=args.top_k,
                    retriever=retriever,
                    dataset=args.dataset,
                    client=client,
                    few_shot=few_shot_samples,
                    upstream_results=upstream_results
                )
                for passage, score in zip(passages, scores):
                    if passage in retrieved_passages_dict:
                        retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                    else:
                        retrieved_passages_dict[passage] = score
            elif node.node_type == NodeType.reasoning:
                result = await reason_step(
                    instruction=node.instruction,
                    client=client,
                    upstream_results=upstream_results
                )
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
            
            thoughts.append(result)
            node_outputs[node.id] = result
            attempt_data["nodes"].append({
                "node_id": node.id,
                "instruction": node.instruction,
                "output": result,
                "node_type": node.node_type.value,
                "upstream_node_ids": node.upstream_node_ids
            })
            
        sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
        attempt_data["retrieved_passages"] = list(retrieved_passages)
        
        if args.dataset in ['hotpotqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif args.dataset in ['musique']:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
            retrieved_items = list(retrieved_passages)
        elif args.dataset in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        
        recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}
        avg_recall = sum(recall.values()) / len(recall)
        attempt_data["recall"] = avg_recall
        
        additional_attempts_best.append(attempt_data)
        
    # save the best graph
    with open(f'result/llm_20/best_graph_additional_{idx}.json', 'w') as f:
        json.dump(additional_attempts_best, f, indent=4)

    # fix the instruction and structure of the fifth, do 10 more retrievals
    additional_attempts_fifth = []
    for i in range(10):
        attempt_data = {
            "attempt": i + 1,
            "question": query,
            "nodes": [],
            "retrieved_passages": [],
            "recall": None
        }
        retrieved_passages_dict = {}
        thoughts = []
        node_outputs = {}

        for node in fifth_graph.graph.nodes:
            upstream_results = [(up_node.id, node_outputs[up_node.id]) for up_node in best_graph.graph.nodes if up_node.id in node.upstream_node_ids]
            if node.node_type == NodeType.retrievalandreasoning:
                result, passages, scores = await retrieve_and_reason_step(
                    query=query,
                    instruction=node.instruction,
                    corpus=corpus,
                    top_k=args.top_k,
                    retriever=retriever,
                    dataset=args.dataset,
                    client=client,
                    few_shot=few_shot_samples,
                    upstream_results=upstream_results
                )
                for passage, score in zip(passages, scores):
                    if passage in retrieved_passages_dict:
                        retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                    else:
                        retrieved_passages_dict[passage] = score
            elif node.node_type == NodeType.reasoning:
                result = await reason_step(
                    instruction=node.instruction,
                    client=client,
                    upstream_results=upstream_results
                )
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")

            thoughts.append(result)
            node_outputs[node.id] = result
            attempt_data["nodes"].append({
                "node_id": node.id,
                "instruction": node.instruction,
                "output": result,
                "node_type": node.node_type.value,
                "upstream_node_ids": node.upstream_node_ids
            })

        sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
        attempt_data["retrieved_passages"] = list(retrieved_passages)

        if args.dataset in ['hotpotqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif args.dataset in ['musique']:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
            retrieved_items = list(retrieved_passages)
        elif args.dataset in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')

        recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}
        avg_recall = sum(recall.values()) / len(recall)
        attempt_data["recall"] = avg_recall
        
        additional_attempts_fifth.append(attempt_data)
        
    # save the fifth graph
    with open(f'result/llm_20/fifth_graph_additional_{idx}.json', 'w') as f:
        json.dump(additional_attempts_fifth, f, indent=4)
        
    # fix the instruction and structure of the lowest, do 10 more retrievals
    additional_attempts_lowest = []
    for i in range(10):
        attempt_data = {
            "attempt": i + 1,
            "question": query,
            "nodes": [],
            "retrieved_passages": [],
            "recall": None
        }
        retrieved_passages_dict = {}
        thoughts = []
        node_outputs = {}

        for node in lowest_graph.graph.nodes:
            upstream_results = [(up_node.id, node_outputs[up_node.id]) for up_node in best_graph.graph.nodes if up_node.id in node.upstream_node_ids]
            if node.node_type == NodeType.retrievalandreasoning:
                result, passages, scores = await retrieve_and_reason_step(
                    query=query,
                    instruction=node.instruction,
                    corpus=corpus,
                    top_k=args.top_k,
                    retriever=retriever,
                    dataset=args.dataset,
                    client=client,
                    few_shot=few_shot_samples,
                    upstream_results=upstream_results
                )
                for passage, score in zip(passages, scores):
                    if passage in retrieved_passages_dict:
                        retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
                    else:
                        retrieved_passages_dict[passage] = score
            elif node.node_type == NodeType.reasoning:
                result = await reason_step(
                    instruction=node.instruction,
                    client=client,
                    upstream_results=upstream_results
                )
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")

            thoughts.append(result)
            node_outputs[node.id] = result
            attempt_data["nodes"].append({
                "node_id": node.id,
                "instruction": node.instruction,
                "output": result,
                "node_type": node.node_type.value,
                "upstream_node_ids": node.upstream_node_ids
            })

        sorted_passages = sorted(retrieved_passages_dict.items(), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages) if sorted_passages else ([], [])
        attempt_data["retrieved_passages"] = list(retrieved_passages)

        if args.dataset in ['hotpotqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        elif args.dataset in ['musique']:
            gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
            gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
            retrieved_items = list(retrieved_passages)
        elif args.dataset in ['2wikimultihopqa']:
            gold_passages = [item for item in sample['supporting_facts']]
            gold_items = set([item[0] for item in gold_passages])
            retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')

        recall = {k: sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items) for k in k_list}
        avg_recall = sum(recall.values()) / len(recall)
        attempt_data["recall"] = avg_recall
        
        additional_attempts_lowest.append(attempt_data)
        
    # save the lowest graph
    with open(f'result/llm_20/lowest_graph_additional_{idx}.json', 'w') as f:
        json.dump(additional_attempts_lowest, f, indent=4)

def visualize_execution_graph(graph_response: GraphResponse):
    G = nx.DiGraph()
    
    node_colors = {
        NodeType.retrieval: 'lightblue',
        NodeType.reasoning: 'lightgreen',
        NodeType.combination: 'lightyellow'
    }
    
    for node in graph_response.graph.nodes:
        G.add_node(node.id, 
                   label=f"Node {node.id}\n{node.node_type.value}\n{node.instruction[:20]}...",
                   color=node_colors[node.node_type])
        for upstream_id in node.upstream_node_ids:
            G.add_edge(upstream_id, node.id)
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    plt.figure(figsize=(15, 10))
    
    nx.draw(G, pos, 
            node_color=[G.nodes[n]['color'] for n in G.nodes()],
            with_labels=False, 
            node_size=3000, 
            font_size=8, 
            font_weight='bold', 
            arrows=True, 
            edge_color='gray')
    
    nx.draw_networkx_labels(G, pos, {node: G.nodes[node]['label'] for node in G.nodes()}, font_size=8)
    
    plt.title("Execution Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
async def main():
    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()
    if read_existing_data:
        print(f'All samples have been already in the result file ({output_path}), exit.')
        exit(0)
    sem = asyncio.Semaphore(100)   # could be larger?
    tasks = []
    import io
    sys.stdout = io.StringIO()
    for idx, sample in enumerate(data):
        async def __task(idx, sample):
            async with sem:
                result = await process_sample(idx, sample, args, corpus, retriever, client, processed_ids)
                if result is not None:
                    idx, recall, retrieved_passages, thoughts, it = result

                    # print metrics
                    for k in k_list:
                        total_recall[k] += recall[k]
                        print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
                    print()
                    if args.max_steps > 1:
                        print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

                    # record results
                    results[idx]['retrieved'] = retrieved_passages
                    results[idx]['recall'] = recall
                    results[idx]['thoughts'] = thoughts

                    # if idx % 50 == 0:
                    #     with open(output_path, 'w') as f:
                    #         json.dump(results, f)
        tasks.append(__task(idx, sample))
    # save final results
    import tqdm.asyncio
    await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved results to {output_path}')
    for k in k_list:
        print(f'R@{k}: {total_recall[k] / len(data):.4f} ', end='')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, choices=['hotpotqa', 'musique', '2wikimultihopqa'], required=True)
    parser.add_argument('--dataset', type=str, default = 'hotpotqa', choices=['hotpotqa', 'musique', '2wikimultihopqa'], required=False)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106')
    # parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--retriever', type=str, default='bm25')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--unit', type=str, choices=['hippo', 'proposition'], default='hippo')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of documents in the demonstration', required=False)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    parser.add_argument('--thread', type=int, default=6, help='number of threads for parallel processing, 1 for sequential processing')
    args = parser.parse_args()

    retriever_name = args.retriever.replace('/', '_').replace('.', '_')
    client = init_langchain_model(args.llm, args.llm_model)
    colbert_configs = {'root': f'data/lm_vectors/colbertv2/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    # Load dataset and corpus
    if args.dataset == 'hotpotqa':
        data = json.load(open('data/hotpotqa.json', 'r'))
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'musique':
        data = json.load(open('data/musique.json', 'r'))
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 4
    elif args.dataset == '2wikimultihopqa':
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    few_shot_samples = parse_prompt(prompt_path)
    few_shot_samples = few_shot_samples[:args.num_demo]
    print('num of demo:', len(few_shot_samples))

    output_path = f'output/graphrag/{args.dataset}_{retriever_name}_demo_{args.num_demo}_{args.llm_model}_step_{max_steps}_top_{args.top_k}.json'

    # Initialize retriever
    if args.retriever == 'bm25':
        if args.unit == 'hippo':
            retriever = BM25Retriever(index_name=f'{args.dataset}_{len(corpus)}_bm25')
        elif args.unit == 'proposition':
            retriever = BM25Retriever(index_name=f'{args.dataset}_{len(corpus)}_proposition_bm25')
    elif args.retriever == 'colbertv2':
        if args.dataset == 'hotpotqa':
            root = 'exp/hotpotqa'
            if args.unit == 'hippo':
                index_name = 'hotpotqa_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = 'hotpotqa_1000_proposition_nbits_2'
        elif args.dataset == 'musique':
            root = 'exp/musique'
            if args.unit == 'hippo':
                index_name = 'musique_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = 'musique_1000_proposition_nbits_2'
        elif args.dataset == '2wikimultihopqa':
            root = 'exp/2wikimultihopqa'
            if args.unit == 'hippo':
                index_name = '2wikimultihopqa_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = '2wikimultihopqa_1000_proposition_nbits_2'
        retriever = Colbertv2Retriever(root, index_name)
    elif args.retriever == 'facebook/contriever':
        if args.dataset == 'hotpotqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_proposition_ip_norm.index')
        elif args.dataset == 'musique':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/musique/musique_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/musique/musique_proposition_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_proposition_ip_norm.index')
        retriever = DPRRetriever(args.retriever, faiss_index, corpus)
    elif args.retriever.startswith('sentence-transformers/gtr-t5'):
        if args.dataset == 'hotpotqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        elif args.dataset == 'musique':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/musique/musique_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/musique/musique_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        retriever = SentenceTransformersRetriever(args.retriever, faiss_index, corpus)

    k_list = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    total_recall = {k: 0 for k in k_list}

    # Read previous results
    results = data
    read_existing_data = False
    processed_ids = set()

    # try:
    #     if os.path.isfile(output_path):
    #         with open(output_path, 'r') as f:
    #             results = json.load(f)
    #             print(f'Loaded {len(results)} results from {output_path}')
    #             if len(results):
    #                 read_existing_data = False # Hard code to turn off checking existing data
    #     if args.dataset in ['hotpotqa', '2wikimultihopqa']:
    #         processed_ids = {sample['_id'] for sample in results if 'retrieved' in sample}
    #     elif args.dataset in ['musique']:
    #         processed_ids = {sample['id'] for sample in results if 'retrieved' in sample}
    #     else:
    #         raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    #     for sample in results:
    #         if 'recall' in sample:
    #             total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
    # except Exception as e:
    #     print('loading results exception', e)
    #     print(f'Results file {output_path} maybe empty, cannot be loaded.')
    #     processed_ids = set()

    asyncio.run(main())