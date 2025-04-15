# api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pickle
import os
from typing import List, Optional, Dict, Any
import uvicorn
from model_classes import LabelPredictor  # Import the class definition

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Use environment variables with fallbacks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "data/label_prediction_model.pkl")
DB_PATH = os.getenv("DB_PATH", "data/mental_health_db.sqlite")


# Load the label prediction model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Counseling Assistant",
    description="API for helping mental health counselors find relevant examples and guidance",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models for API endpoints
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class LabelPredictionResponse(BaseModel):
    issues: List[Dict[str, Any]]
    response_types: List[Dict[str, Any]]
    approaches: List[Dict[str, Any]]

class ConversationResponse(BaseModel):
    conversation_id: int
    context: str
    response: str
    similarity: float
    issues: Optional[List[str]] = None
    response_types: Optional[List[str]] = None
    approaches: Optional[List[str]] = None

class SearchResponse(BaseModel):
    predictions: LabelPredictionResponse
    results: List[ConversationResponse]

# Define API endpoints
@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "ok", "message": "Mental Health Counseling Assistant API is running"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for relevant mental health conversations based on a query"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Use our label prediction model to search
        results = model.search(request.query, top_k=request.limit)
        
        # Format the response
        formatted_results = []
        for result in results["results"]:
            formatted_results.append(
                ConversationResponse(
                    conversation_id=result["conversation_id"],
                    context=result["context"],
                    response=result["response"],
                    similarity=result["similarity"],
                    # We could add issues, response_types, approaches here if needed
                )
            )
        
        # Format the predictions
        predictions = {
            "issues": [{"label": issue, "confidence": score} for issue, score in results["predictions"]["issues"]],
            "response_types": [{"label": rt, "confidence": score} for rt, score in results["predictions"]["response_types"]],
            "approaches": [{"label": approach, "confidence": score} for approach, score in results["predictions"]["approaches"]],
        }
        
        return {
            "predictions": predictions,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.get("/labels")
async def get_labels():
    """Get all available labels for filtering"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all issues
        cursor.execute("SELECT issue_name FROM issues ORDER BY issue_name")
        issues = [row[0] for row in cursor.fetchall()]
        
        # Get all response types
        cursor.execute("SELECT response_type_name FROM response_types ORDER BY response_type_name")
        response_types = [row[0] for row in cursor.fetchall()]
        
        # Get all therapeutic approaches
        cursor.execute("SELECT approach_name FROM therapeutic_approaches ORDER BY approach_name")
        approaches = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "issues": issues,
            "response_types": response_types,
            "approaches": approaches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get a specific conversation by ID with all its labels"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        
        # Get the conversation with its labels
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
        WHERE c.conversation_id = ?
        GROUP BY c.conversation_id
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (conversation_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
        
        # Parse the results
        issues = result[3].split(',') if result[3] else []
        response_types = result[4].split(',') if result[4] else []
        approaches = result[5].split(',') if result[5] else []
        
        return {
            "conversation_id": result[0],
            "context": result[1],
            "response": result[2],
            "issues": issues,
            "response_types": response_types,
            "approaches": approaches
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/feedback")
async def record_feedback(
    conversation_id: int,
    helpful: bool,
    comments: Optional[str] = None
):
    """Record user feedback on a conversation"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if feedback table exists, create if not
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            helpful BOOLEAN,
            comments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
        )
        """)
        
        # Insert the feedback
        cursor.execute(
            "INSERT INTO feedback (conversation_id, helpful, comments) VALUES (?, ?, ?)",
            (conversation_id, helpful, comments)
        )
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)