// src/components/SearchBar.jsx
import React, { useState } from "react";

const SearchBar = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query);
    }
  };

  return (
    <div className="w-full mx-auto mb-8 bg-white p-6 rounded-lg shadow-sm border border-slate-200">
      <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-3">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Describe your patient's situation or the guidance you're seeking..."
        className="flex-grow p-3 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-black"
        disabled={isLoading}
      />
        <button
          type="submit"
          className={`px-5 py-3 rounded-lg font-medium transition-colors bg-slate-700 hover:bg-slate-800 ${
            isLoading
              ? "bg-slate-400 cursor-not-allowed text-white"
              : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
          disabled={isLoading}
        >
          {isLoading ? "Searching..." : "Search"}
        </button>
      </form>
      <p className="mt-3 text-sm text-slate-500">
        Example: "Patient experiencing panic attacks at work" or "How to respond to grief over loss of spouse"
      </p>
    </div>
  );
};

export default SearchBar;