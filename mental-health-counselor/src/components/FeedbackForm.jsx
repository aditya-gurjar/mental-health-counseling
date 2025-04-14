// src/components/FeedbackForm.jsx
import React, { useState } from "react";
import { submitFeedback } from "../api";

const FeedbackForm = ({ conversationId, onSubmit }) => {
  const [helpful, setHelpful] = useState(null);
  const [comments, setComments] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (helpful === null) {
      setError("Please select whether this was helpful");
      return;
    }
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      await submitFeedback(conversationId, helpful, comments);
      onSubmit();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="mt-4 pt-4 border-t border-slate-100">
      <h4 className="text-sm font-medium text-slate-700 mb-3">Was this example helpful?</h4>
      
      <div className="flex gap-4 mb-3">
        <button
          type="button"
          onClick={() => setHelpful(true)}
          className={`px-4 py-1.5 text-sm rounded-full flex items-center ${
            helpful === true
              ? "bg-green-100 text-green-800 border border-green-300"
              : "bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200"
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905c0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
          </svg>
          Yes
        </button>
        <button
          type="button"
          onClick={() => setHelpful(false)}
          className={`px-4 py-1.5 text-sm rounded-full flex items-center ${
            helpful === false
              ? "bg-red-100 text-red-800 border border-red-300"
              : "bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200"
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018c.163 0 .326.02.485.06L17 4m-10 10v7m0-7H7m10 7h2a2 2 0 002-2v-6a2 2 0 00-2-2h-2.5" />
          </svg>
          No
        </button>
      </div>
      
      <textarea
        placeholder="Optional comments on why this was or wasn't helpful..."
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        className="w-full p-3 text-sm border border-slate-300 rounded-md mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        rows={2}
      />
      
      {error && <p className="text-red-600 text-xs mb-2">{error}</p>}
      
      <button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className={`px-4 py-2 text-sm rounded-md font-medium transition-colors ${
          isSubmitting
            ? "bg-slate-300 text-slate-500 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 text-white"
        }`}
      >
        {isSubmitting ? "Submitting..." : "Submit Feedback"}
      </button>
    </div>
  );
};

export default FeedbackForm;