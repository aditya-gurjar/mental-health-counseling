import pandas as pd
import sqlite3
import json
import numpy as np
import pickle
from tqdm import tqdm
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import time
import tiktoken
from openai import OpenAI

# Load environment variables (API keys)
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def num_tokens_from_string(string, model="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_data_from_database(db_path):
    """Load the conversations and their labels from the database"""
    print(f"Loading data from database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    
    # Get conversations with their labels
    query = """
    SELECT 
        c.conversation_id, 
        c.context, 
        c.response,
        GROUP_CONCAT(DISTINCT i.issue_name) as issues,
        GROUP_CONCAT(DISTINCT rt.response_type_name) as response_types,
        GROUP_CONCAT(DISTINCT a.approach_name) as approaches
    FROM conversations c
    LEFT JOIN conversation_issues ci ON c.conversation_id = ci.conversation_id
    LEFT JOIN issues i ON ci.issue_id = i.issue_id
    LEFT JOIN conversation_response_types crt ON c.conversation_id = crt.conversation_id
    LEFT JOIN response_types rt ON crt.response_type_id = rt.response_type_id
    LEFT JOIN conversation_approaches ca ON c.conversation_id = ca.conversation_id
    LEFT JOIN therapeutic_approaches a ON ca.approach_id = a.approach_id
    GROUP BY c.conversation_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert comma-separated strings to lists (SQLite's default separator for GROUP_CONCAT is comma)
    df['issues'] = df['issues'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['response_types'] = df['response_types'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df['approaches'] = df['approaches'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    
    print(f"Loaded {len(df)} conversations with their labels")
    return df

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

def generate_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    """Generate embeddings for a batch of texts using updated OpenAI API"""
    all_embeddings = []
    
    # First check token counts to avoid API errors
    token_counts = [num_tokens_from_string(text) for text in texts]
    over_limit_indices = [i for i, count in enumerate(token_counts) if count > 8191]
    
    if over_limit_indices:
        print(f"Warning: {len(over_limit_indices)} texts exceed the token limit (8191)")
        # Truncate texts that are too long (you might want to handle this differently)
        for i in over_limit_indices:
            texts[i] = texts[i][:8000]  # Simple truncation
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            # Clean texts
            batch_clean = [text.replace("\n", " ") for text in batch]
            
            # Get embeddings for batch
            response = client.embeddings.create(
                input=batch_clean,
                model=model
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Sleep to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error getting embeddings for batch starting at {i}: {e}")
            # On error, fill with empty embeddings
            empty_embeddings = [None] * len(batch)
            all_embeddings.extend(empty_embeddings)
    
    return all_embeddings

def create_conversation_embeddings(df, output_path):
    """Create and save embeddings for all conversations"""
    print("Generating embeddings for conversations...")
    
    # Combine context and response for each conversation
    combined_texts = [f"{row['context']} {row['response']}" for _, row in df.iterrows()]
    
    # Generate embeddings in batches
    embeddings = generate_embeddings_batch(combined_texts, batch_size=100)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Filter out any None embeddings
    df_valid = df.dropna(subset=['embedding'])
    print(f"Generated {len(df_valid)} valid embeddings out of {len(df)} conversations")
    
    # Save embeddings to file
    with open(output_path, 'wb') as f:
        pickle.dump({
            'conversation_ids': df_valid['conversation_id'].tolist(),
            'embeddings': df_valid['embedding'].tolist()
        }, f)
    
    print(f"Saved embeddings to {output_path}")
    return df_valid

def create_inverted_index(df):
    """Create an inverted index mapping labels to conversation IDs"""
    print("Creating inverted index...")
    
    # Initialize dictionaries for each label type
    issue_index = {}
    response_type_index = {}
    approach_index = {}
    
    # Build the indices
    for _, row in tqdm(df.iterrows(), total=len(df)):
        conv_id = row['conversation_id']
        
        # Add to issue index
        for issue in row['issues']:
            if issue not in issue_index:
                issue_index[issue] = []
            issue_index[issue].append(conv_id)
        
        # Add to response type index
        for response_type in row['response_types']:
            if response_type not in response_type_index:
                response_type_index[response_type] = []
            response_type_index[response_type].append(conv_id)
        
        # Add to approach index
        for approach in row['approaches']:
            if approach not in approach_index:
                approach_index[approach] = []
            approach_index[approach].append(conv_id)
    
    # Create the complete inverted index
    inverted_index = {
        'issues': issue_index,
        'response_types': response_type_index,
        'approaches': approach_index
    }
    
    print(f"Created inverted index with {len(issue_index)} issues, " 
          f"{len(response_type_index)} response types, and {len(approach_index)} approaches")
    
    return inverted_index

def save_inverted_index(inverted_index, output_path):
    """Save the inverted index to a file"""
    with open(output_path, 'wb') as f:
        pickle.dump(inverted_index, f)
    
    print(f"Saved inverted index to {output_path}")
    return True

def create_label_prediction_function(df_with_embeddings, inverted_index):
    """Create a function to predict labels from search queries using embeddings"""
    
    def predict_labels_for_query(query_text, top_k=5):
        # Get embedding for the query
        try:
            query_clean = query_text.replace("\n", " ")
            response = client.embeddings.create(
                input=[query_clean],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return None
        
        # Calculate similarity with all conversation embeddings
        similarities = []
        for idx, row in df_with_embeddings.iterrows():
            conv_embedding = row['embedding']
            sim = cosine_similarity([query_embedding], [conv_embedding])[0][0]
            similarities.append((row['conversation_id'], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K most similar conversations
        top_conversations = similarities[:top_k]
        
        # Extract and count labels from these conversations
        issue_counts = {}
        response_type_counts = {}
        approach_counts = {}
        
        for conv_id, sim in top_conversations:
            # Find the conversation in the dataframe
            conv_row = df_with_embeddings[df_with_embeddings['conversation_id'] == conv_id].iloc[0]
            
            # Count issues
            for issue in conv_row['issues']:
                if issue not in issue_counts:
                    issue_counts[issue] = 0
                issue_counts[issue] += sim  # Weight by similarity
            
            # Count response types
            for response_type in conv_row['response_types']:
                if response_type not in response_type_counts:
                    response_type_counts[response_type] = 0
                response_type_counts[response_type] += sim  # Weight by similarity
            
            # Count approaches
            for approach in conv_row['approaches']:
                if approach not in approach_counts:
                    approach_counts[approach] = 0
                approach_counts[approach] += sim  # Weight by similarity
        
        # Sort by count
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_response_types = sorted(response_type_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_approaches = sorted(approach_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top labels (all with scores > 0)
        predicted_issues = [issue for issue, count in sorted_issues if count > 0]
        predicted_response_types = [rt for rt, count in sorted_response_types if count > 0]
        predicted_approaches = [approach for approach, count in sorted_approaches if count > 0]
        
        return {
            'issues': predicted_issues,
            'response_types': predicted_response_types,
            'approaches': predicted_approaches
        }
    
    return predict_labels_for_query

def test_search_system(df, inverted_index, predict_labels_function):
    """Test the search system with a few examples"""
    print("\nTesting search system...")
    
    test_queries = [
        "patient with anxiety about work performance",
        "teen dealing with depression and self-harm",
        "struggling with grief after losing a parent",
        "addiction to prescription medication"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # 1. Predict labels
        predicted_labels = predict_labels_function(query)
        print(f"Predicted issues: {predicted_labels['issues'][:3]}")
        print(f"Predicted response types: {predicted_labels['response_types'][:3]}")
        
        # 2. Use inverted index to find relevant conversations
        relevant_conv_ids = set()
        
        # Add conversations matching predicted issues
        for issue in predicted_labels['issues'][:3]:  # Use top 3 issues
            if issue in inverted_index['issues']:
                relevant_conv_ids.update(inverted_index['issues'][issue])
        
        # Add conversations matching predicted response types (optional)
        for response_type in predicted_labels['response_types'][:2]:  # Use top 2 response types
            if response_type in inverted_index['response_types']:
                relevant_conv_ids.update(inverted_index['response_types'][response_type])
        
        print(f"Found {len(relevant_conv_ids)} potentially relevant conversations")
        
        # 3. Rank by embedding similarity if we have a function for it
        if len(relevant_conv_ids) > 0:
            # Just show the first result for testing
            sample_conv_id = list(relevant_conv_ids)[0]
            sample_conv = df[df['conversation_id'] == sample_conv_id].iloc[0]
            
            print(f"Sample conversation (ID: {sample_conv_id}):")
            print(f"Patient: {sample_conv['context'][:100]}...")
            print(f"Therapist: {sample_conv['response'][:100]}...")
            print(f"Issues: {sample_conv['issues']}")
            print(f"Response types: {sample_conv['response_types']}")
    
    return True

def main():
    # Set file paths
    db_path = "data/mental_health_db.sqlite"
    embeddings_path = "data/conversation_embeddings.pkl"
    inverted_index_path = "data/inverted_index.pkl"
    
    # Load data from database
    df = load_data_from_database(db_path)
    
    # Create embeddings
    df_with_embeddings = create_conversation_embeddings(df, embeddings_path)
    
    # Create inverted index
    inverted_index = create_inverted_index(df)
    
    # Save inverted index
    save_inverted_index(inverted_index, inverted_index_path)
    
    # Create label prediction function
    predict_labels_function = create_label_prediction_function(df_with_embeddings, inverted_index)
    
    # Test the search system
    test_search_system(df, inverted_index, predict_labels_function)
    
    print("\nSearch system setup completed successfully!")

if __name__ == "__main__":
    main()