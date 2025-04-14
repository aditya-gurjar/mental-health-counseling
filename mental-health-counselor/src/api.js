const API_URL = "http://localhost:8000"; // Change in production

export const searchConversations = async (query, limit = 5) => {
  const response = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, limit }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to search conversations");
  }

  return response.json();
};

export const getLabels = async () => {
  const response = await fetch(`${API_URL}/labels`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to fetch labels");
  }

  return response.json();
};

export const getConversation = async (conversationId) => {
  const response = await fetch(`${API_URL}/conversations/${conversationId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to fetch conversation");
  }

  return response.json();
};

export const submitFeedback = async (conversationId, helpful, comments) => {
  const response = await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ conversation_id: conversationId, helpful, comments }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to submit feedback");
  }

  return response.json();
};