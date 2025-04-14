# model_classes.py
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

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
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=[text],
                model=model
            )
            embedding = response.data[0].embedding
            client.close()
            return embedding
        except Exception as e:
            print(f"Error getting embedding in model_classes.py: {e}")
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