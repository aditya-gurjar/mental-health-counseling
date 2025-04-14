import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from tqdm import tqdm
import tiktoken

# Load environment variables
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instead of trying to pickle the functions directly, let's create a class
# that encapsulates the functionality

class LabelPredictor:
    def __init__(self, conversation_ids, embeddings, inverted_index, conversations):
        # Store all the data needed for prediction
        self.conversation_ids = conversation_ids
        self.embeddings = embeddings
        self.inverted_index = inverted_index
        self.conversations = conversations
        
        # Create mapping from conversation IDs to embeddings
        self.embedding_map = dict(zip(conversation_ids, embeddings))
        
        # Create mapping from conversation IDs to row indices
        self.id_to_index = {conv_id: i for i, conv_id in enumerate(conversations['conversation_id'])}
        
        # Cache for query embeddings
        self.query_embedding_cache = {}
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """Get embeddings for a single text using OpenAI's API"""
        try:
            text = text.replace("\n", " ")
            response = client.embeddings.create(
                input=[text],
                model=model
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def predict_labels(self, query_text, top_k=5, threshold=0.5):
        """Predict labels for a search query"""
        # Check cache first for query embedding
        if query_text in self.query_embedding_cache:
            query_embedding = self.query_embedding_cache[query_text]
        else:
            # Get embedding for the query
            query_embedding = self.get_embedding(query_text)
            if query_embedding:
                # Cache the embedding for future use
                self.query_embedding_cache[query_text] = query_embedding
        
        if not query_embedding:
            return None
        
        # Find similar conversations using embeddings
        similarities = []
        for conv_id in self.conversation_ids:
            if conv_id in self.embedding_map:
                sim = cosine_similarity([query_embedding], [self.embedding_map[conv_id]])[0][0]
                similarities.append((conv_id, sim))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_conversations = similarities[:top_k]
        
        # Track label frequencies weighted by similarity
        issue_scores = {}
        response_type_scores = {}
        approach_scores = {}
        
        # Get labels for each conversation through the inverted index
        for conv_id, sim in top_conversations:
            # Only consider conversations with similarity above threshold
            if sim < threshold:
                continue
                
            # Find issues for this conversation
            for issue, conv_ids in self.inverted_index['issues'].items():
                if conv_id in conv_ids:
                    if issue not in issue_scores:
                        issue_scores[issue] = 0
                    issue_scores[issue] += sim
            
            # Find response types for this conversation
            for resp_type, conv_ids in self.inverted_index['response_types'].items():
                if conv_id in conv_ids:
                    if resp_type not in response_type_scores:
                        response_type_scores[resp_type] = 0
                    response_type_scores[resp_type] += sim
            
            # Find therapeutic approaches for this conversation
            for approach, conv_ids in self.inverted_index['approaches'].items():
                if conv_id in conv_ids:
                    if approach not in approach_scores:
                        approach_scores[approach] = 0
                    approach_scores[approach] += sim
        
        # Sort by score
        sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_response_types = sorted(response_type_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_approaches = sorted(approach_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top predictions (label and normalized confidence scores)
        total_issue_score = sum(issue_scores.values()) if issue_scores else 1
        total_response_score = sum(response_type_scores.values()) if response_type_scores else 1
        total_approach_score = sum(approach_scores.values()) if approach_scores else 1
        
        top_issues = [(issue, score/total_issue_score) for issue, score in sorted_issues]
        top_response_types = [(rt, score/total_response_score) for rt, score in sorted_response_types]
        top_approaches = [(approach, score/total_approach_score) for approach, score in sorted_approaches]
        
        # Return predictions with confidence scores
        predictions = {
            'issues': top_issues[:5],  # Return top 5 with scores
            'response_types': top_response_types[:3],  # Return top 3 with scores
            'approaches': top_approaches[:3]  # Return top 3 with scores
        }
        
        # Also return the top similar conversations for reference
        top_conv_details = []
        for conv_id, sim in top_conversations[:3]:
            if conv_id in self.id_to_index:
                idx = self.id_to_index[conv_id]
                conv = self.conversations.iloc[idx]
                top_conv_details.append({
                    'id': conv_id,
                    'similarity': sim,
                    'context': conv['context'][:200] + "...",  # Preview
                    'response': conv['response'][:200] + "..."  # Preview
                })
        
        predictions['similar_conversations'] = top_conv_details
        
        return predictions
    
    def search(self, query, top_k=5):
        """Complete search function that returns the most relevant conversations"""
        # Step 1: Predict labels from query
        predictions = self.predict_labels(query)
        if not predictions:
            return {"error": "Failed to get predictions for query"}
        
        # Extract just the labels (without scores) for filtering
        predicted_issues = [issue for issue, score in predictions['issues']]
        predicted_response_types = [rt for rt, score in predictions['response_types']]
        
        # Step 2: Get query embedding for semantic search
        if query in self.query_embedding_cache:
            query_embedding = self.query_embedding_cache[query]
        else:
            query_embedding = self.get_embedding(query)
            if query_embedding:
                self.query_embedding_cache[query] = query_embedding
        
        if not query_embedding:
            return {"error": "Failed to get embedding for query"}
        
        # Step 3: Get all conversations with the predicted labels
        relevant_conversations = set()
        
        # Get conversations from predicted issues (if any)
        for issue in predicted_issues:
            if issue in self.inverted_index['issues']:
                relevant_conversations.update(self.inverted_index['issues'][issue])
        
        # Get conversations from predicted response types (if any)
        for rt in predicted_response_types:
            if rt in self.inverted_index['response_types']:
                relevant_conversations.update(self.inverted_index['response_types'][rt])
        
        # If we don't have enough conversations yet, include all
        if len(relevant_conversations) < 10:
            relevant_conversations = set(self.conversation_ids)
        
        # Step 4: Rank the filtered conversations by embedding similarity
        ranked_results = []
        for conv_id in relevant_conversations:
            if conv_id in self.embedding_map:
                # Calculate similarity
                similarity = cosine_similarity([query_embedding], [self.embedding_map[conv_id]])[0][0]
                
                # Find conversation details
                conv = self.conversations[self.conversations['conversation_id'] == conv_id]
                if not conv.empty:
                    conv_data = conv.iloc[0]
                    ranked_results.append({
                        'conversation_id': conv_id,
                        'similarity': similarity,
                        'context': conv_data['context'],
                        'response': conv_data['response']
                    })
        
        # Sort by similarity
        ranked_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        return {
            "predictions": predictions,
            "results": ranked_results[:top_k]
        }

def num_tokens_from_string(string, model="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_embeddings(embeddings_path):
    """Load precomputed embeddings from file"""
    print(f"Loading embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    conversation_ids = embeddings_data['conversation_ids']
    embeddings = embeddings_data['embeddings']
    
    print(f"Loaded {len(conversation_ids)} embeddings")
    return conversation_ids, embeddings

def load_inverted_index(index_path):
    """Load inverted index from file"""
    print(f"Loading inverted index from {index_path}...")
    with open(index_path, 'rb') as f:
        inverted_index = pickle.load(f)
    
    issue_count = len(inverted_index['issues'])
    response_type_count = len(inverted_index['response_types'])
    approach_count = len(inverted_index['approaches'])
    
    print(f"Loaded index with {issue_count} issues, {response_type_count} response types, and {approach_count} approaches")
    return inverted_index

def load_conversation_data(db_path):
    """Load conversation data from the database"""
    import sqlite3
    
    print(f"Loading conversation data from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Query to get just the conversation content
    query = """
    SELECT 
        c.conversation_id, 
        c.context, 
        c.response 
    FROM conversations c
    """
    
    conversations = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(conversations)} conversations")
    return conversations

def get_embedding(text, model="text-embedding-3-small"):
    """Get embeddings for a single text using OpenAI's API"""
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def create_label_prediction_model(conversation_ids, embeddings, inverted_index, conversations):
    """Create a model to predict labels from search queries based on embeddings"""
    
    # Create a dictionary mapping conversation IDs to their embeddings
    embedding_map = dict(zip(conversation_ids, embeddings))
    
    # Create a mapping from conversation IDs to row indices
    id_to_index = {conv_id: i for i, conv_id in enumerate(conversations['conversation_id'])}
    
    # Create reverse mappings for labels to make prediction easier
    all_issues = list(inverted_index['issues'].keys())
    all_response_types = list(inverted_index['response_types'].keys())
    all_approaches = list(inverted_index['approaches'].keys())
    
    print(f"Model will predict from {len(all_issues)} issues, " 
          f"{len(all_response_types)} response types, and {len(all_approaches)} approaches")
    
    def predict_labels(query_text, top_k=5, threshold=0.5):
        """Predict labels for a search query"""
        # Get embedding for the query
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            return None
        
        # Find similar conversations using embeddings
        similarities = []
        for conv_id in conversation_ids:
            if conv_id in embedding_map:
                sim = cosine_similarity([query_embedding], [embedding_map[conv_id]])[0][0]
                similarities.append((conv_id, sim))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_conversations = similarities[:top_k]
        
        # Track label frequencies weighted by similarity
        issue_scores = {}
        response_type_scores = {}
        approach_scores = {}
        
        # Get labels for each conversation through the inverted index
        for conv_id, sim in top_conversations:
            # Only consider conversations with similarity above threshold
            if sim < threshold:
                continue
                
            # Find issues for this conversation
            for issue, conv_ids in inverted_index['issues'].items():
                if conv_id in conv_ids:
                    if issue not in issue_scores:
                        issue_scores[issue] = 0
                    issue_scores[issue] += sim
            
            # Find response types for this conversation
            for resp_type, conv_ids in inverted_index['response_types'].items():
                if conv_id in conv_ids:
                    if resp_type not in response_type_scores:
                        response_type_scores[resp_type] = 0
                    response_type_scores[resp_type] += sim
            
            # Find therapeutic approaches for this conversation
            for approach, conv_ids in inverted_index['approaches'].items():
                if conv_id in conv_ids:
                    if approach not in approach_scores:
                        approach_scores[approach] = 0
                    approach_scores[approach] += sim
        
        # Sort by score
        sorted_issues = sorted(issue_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_response_types = sorted(response_type_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_approaches = sorted(approach_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top predictions (label and normalized confidence scores)
        top_issues = [(issue, score/sum(issue_scores.values()) if issue_scores else 0) 
                      for issue, score in sorted_issues]
        
        top_response_types = [(rt, score/sum(response_type_scores.values()) if response_type_scores else 0) 
                              for rt, score in sorted_response_types]
        
        top_approaches = [(approach, score/sum(approach_scores.values()) if approach_scores else 0) 
                         for approach, score in sorted_approaches]
        
        # Return predictions with confidence scores
        predictions = {
            'issues': top_issues[:5],  # Return top 5 with scores
            'response_types': top_response_types[:3],  # Return top 3 with scores
            'approaches': top_approaches[:3]  # Return top 3 with scores
        }
        
        # Also return the top similar conversations for reference
        top_conv_details = []
        for conv_id, sim in top_conversations[:3]:
            if conv_id in id_to_index:
                idx = id_to_index[conv_id]
                conv = conversations.iloc[idx]
                top_conv_details.append({
                    'id': conv_id,
                    'similarity': sim,
                    'context': conv['context'][:200] + "...",  # Preview
                    'response': conv['response'][:200] + "..."  # Preview
                })
        
        predictions['similar_conversations'] = top_conv_details
        
        return predictions
    
    return predict_labels

def create_search_function(predict_labels_func, conversation_ids, embeddings, conversations):
    """Create a complete search function that uses label prediction and similarity"""
    
    # Create a dictionary mapping conversation IDs to their embeddings
    embedding_map = dict(zip(conversation_ids, embeddings))
    
    def search(query, top_k=5):
        """Complete search function that returns the most relevant conversations"""
        # Step 1: Predict labels from query
        predictions = predict_labels_func(query)
        if not predictions:
            return {"error": "Failed to get predictions for query"}
        
        # Extract just the labels (without scores) for filtering
        predicted_issues = [issue for issue, score in predictions['issues']]
        predicted_response_types = [rt for rt, score in predictions['response_types']]
        
        # Step 2: Get query embedding for semantic search
        query_embedding = get_embedding(query)
        if not query_embedding:
            return {"error": "Failed to get embedding for query"}
        
        # Step 3: Get all conversations with the predicted labels
        relevant_conversations = set()
        
        # Get conversations from predicted issues (if any)
        for conv_id in conversations['conversation_id']:
            # Check if the conversation has relevant issues or response types
            matches_issue = any(conv_id in embedding_map for issue in predicted_issues)
            matches_response = any(conv_id in embedding_map for rt in predicted_response_types)
            
            if matches_issue or matches_response:
                relevant_conversations.add(conv_id)
        
        # If we don't have enough conversations yet, include all
        if len(relevant_conversations) < 10:
            relevant_conversations = set(conversation_ids)
        
        # Step 4: Rank the filtered conversations by embedding similarity
        ranked_results = []
        for conv_id in relevant_conversations:
            if conv_id in embedding_map:
                # Calculate similarity
                similarity = cosine_similarity([query_embedding], [embedding_map[conv_id]])[0][0]
                
                # Find conversation details
                conv = conversations[conversations['conversation_id'] == conv_id]
                if not conv.empty:
                    conv_data = conv.iloc[0]
                    ranked_results.append({
                        'conversation_id': conv_id,
                        'similarity': similarity,
                        'context': conv_data['context'],
                        'response': conv_data['response']
                    })
        
        # Sort by similarity
        ranked_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top results
        return {
            "predictions": predictions,
            "results": ranked_results[:top_k]
        }
    
    return search

def test_label_prediction_model(model):
    """Test the label prediction model with sample queries"""
    print("\nTesting label prediction model...")
    
    test_queries = [
        "helping a patient with severe anxiety about public speaking",
        "teenager expressing suicidal thoughts after breakup",
        "client grieving the loss of a parent",
        "patient struggling with work-life balance and burnout"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        predictions = model.predict_labels(query)
        
        print("Predicted issues:")
        for issue, score in predictions['issues'][:3]:
            print(f"  - {issue} (confidence: {score:.2f})")
        
        print("Predicted response types:")
        for rt, score in predictions['response_types'][:2]:
            print(f"  - {rt} (confidence: {score:.2f})")
        
        print("Sample similar conversation:")
        if predictions['similar_conversations']:
            sample = predictions['similar_conversations'][0]
            print(f"  ID: {sample['id']}, Similarity: {sample['similarity']:.2f}")
            print(f"  Patient: {sample['context'][:100]}...")
    
    return True

def test_search_function(model):
    """Test the complete search function"""
    print("\nTesting complete search function...")
    
    test_queries = [
        "therapist advice for patient with depression and anxiety",
        "how to respond to client experiencing panic attacks",
        "counseling approach for grief and loss"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        search_results = model.search(query)
        
        print("Predicted labels:")
        for issue, score in search_results["predictions"]["issues"][:3]:
            print(f"  - {issue} (confidence: {score:.2f})")
        
        print("Top results:")
        for i, result in enumerate(search_results["results"][:2]):
            print(f"Result {i+1} (similarity: {result['similarity']:.2f}):")
            print(f"  Patient: {result['context'][:100]}...")
            print(f"  Therapist: {result['response'][:100]}...")
    
    return True


def main():
    # Set file paths
    db_path = "data/mental_health_db.sqlite"
    embeddings_path = "data/conversation_embeddings.pkl"
    inverted_index_path = "data/inverted_index.pkl"
    model_output_path = "data/label_prediction_model.pkl"
    
    # Load data
    conversation_ids, embeddings = load_embeddings(embeddings_path)
    inverted_index = load_inverted_index(inverted_index_path)
    conversations = load_conversation_data(db_path)
    
    # Create the label prediction model (as a class instance)
    model = LabelPredictor(conversation_ids, embeddings, inverted_index, conversations)
    
    # Test the label prediction model
    test_label_prediction_model(model)
    
    # Test the search function
    test_search_function(model)
    
    # Save the model
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nSaved label prediction model to {model_output_path}")
    print("Label prediction model setup completed successfully!")

if __name__ == "__main__":
    main()