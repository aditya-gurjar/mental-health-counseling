// src/components/ResultsList.jsx
import React from "react";
import ConversationCard from "./ConversationCard";

const ResultsList = ({ results, isLoading }) => {
  if (isLoading) {
    return (
      <div className="text-center py-12 bg-white rounded-lg shadow-sm border border-slate-200">
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
        <p className="text-slate-600">Searching for relevant conversations...</p>
        <p className="text-slate-500 text-sm mt-2">
          This may take a moment as we analyze your query and find the best matches.
        </p>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="text-center py-10 bg-white rounded-lg shadow-sm border border-slate-200">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-slate-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-slate-700 font-medium mb-2">No relevant conversations found</p>
        <p className="text-sm text-slate-500 max-w-md mx-auto">
          Try using different keywords, being more specific about the patient's condition, 
          or describing the therapeutic approach you're looking for.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-5 text-slate-800 flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        {results.length} Relevant Conversations
      </h2>
      <div className="space-y-5">
        {results.map((conversation) => (
          <ConversationCard
            key={conversation.conversation_id}
            conversation={conversation}
          />
        ))}
      </div>
    </div>
  );
};

export default ResultsList;