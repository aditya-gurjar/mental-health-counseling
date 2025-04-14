// src/components/ConversationCard.jsx
import React, { useState } from "react";
// import FeedbackForm from "./FeedbackForm";

const ConversationCard = ({ conversation }) => {
  const [expanded, setExpanded] = useState(false);
  // const [showFeedback, setShowFeedback] = useState(false);
  // const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const toggleExpand = () => setExpanded(!expanded);
  const relevancePercent = Math.round(conversation.similarity * 100);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 mb-5">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-semibold text-slate-800">Patient Conversation</h3>
        <span className={`text-sm px-3 py-1 rounded-full font-medium ${
          relevancePercent > 70 
            ? "bg-green-100 text-green-800 border border-green-200" 
            : "bg-slate-100 text-slate-700 border border-slate-200"
        }`}>
          Relevance: {relevancePercent}%
        </span>
      </div>

      <div className="mb-4">
        <div className="flex items-center text-sm font-medium text-slate-600 mb-2">
          <span className="mr-2">🧑‍⚕️</span>
          <span>Patient:</span>
        </div>
        <p className="text-slate-800 bg-slate-50 p-4 rounded-md border border-slate-100">
          {expanded
            ? conversation.context
            : `${conversation.context.slice(0, 200)}${
                conversation.context.length > 200 ? "..." : ""
              }`}
        </p>
      </div>

      <div className="mb-5">
        <div className="flex items-center text-sm font-medium text-slate-600 mb-2">
          <span className="mr-2">👨‍⚕️</span>
          <span>Therapist:</span>
        </div>
        <p className="text-slate-800 bg-blue-50 p-4 rounded-md border border-blue-100">
          {expanded
            ? conversation.response
            : `${conversation.response.slice(0, 200)}${
                conversation.response.length > 200 ? "..." : ""
              }`}
        </p>
      </div>

      <div className="flex justify-between items-center mt-5 pt-3 border-t border-slate-100">
      <button
        onClick={toggleExpand}
        className="text-sm font-medium text-white bg-slate-700 hover:bg-slate-800 border border-blue-200 rounded-md px-4 py-2 flex items-center gap-1 transition"
      >
        {expanded ? (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
            Show Less
          </>
        ) : (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            Show More
          </>
        )}
      </button>

        {/* {!feedbackSubmitted && (
          <button
            onClick={() => setShowFeedback(!showFeedback)}
            className="text-slate-600 hover:text-slate-800 text-sm font-medium hover:underline flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905c0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
            </svg>
            {showFeedback ? "Cancel Feedback" : "Was This Helpful?"}
          </button>
        )} */}
      </div>

      {/* {showFeedback && (
        <FeedbackForm
          conversationId={conversation.conversation_id}
          onSubmit={() => {
            setShowFeedback(false);
            setFeedbackSubmitted(true);
          }}
        />
      )} */}

      {/* {feedbackSubmitted && (
        <div className="mt-3 text-sm text-green-600 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Thank you for your feedback!
        </div>
      )} */}
    </div>
  );
};

export default ConversationCard;