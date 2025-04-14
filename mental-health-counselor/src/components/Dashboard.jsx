// src/components/Dashboard.jsx
import React, { useState, useEffect } from "react";
import SearchBar from "./SearchBar";
import LabelDisplay from "./LabelDisplay";
import ResultsList from "./ResultsList";
import { searchConversations, getLabels } from "../api";

const Dashboard = () => {
  const [searchResults, setSearchResults] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [labels, setLabels] = useState(null);

  useEffect(() => {
    // Load available labels on component mount
    const loadLabels = async () => {
      try {
        const data = await getLabels();
        setLabels(data);
      } catch (err) {
        console.error("Failed to load labels:", err);
      }
    };

    loadLabels();
  }, []);

  const handleSearch = async (query) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await searchConversations(query);
      setSearchResults(data.results);
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message);
      setSearchResults(null);
      setPredictions(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="bg-slate-800 text-white py-10 px-6 shadow-md mb-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold mb-4 text-center">
            Mental Health Counseling Assistant
          </h1>
          <p className="text-slate-200 text-center max-w-3xl mx-auto">
            Search our database of counseling conversations to find examples and
            guidance for your specific patient scenarios.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 pb-12">
        <SearchBar onSearch={handleSearch} isLoading={isLoading} />

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md mb-6">
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        )}

        {predictions && <LabelDisplay predictions={predictions} />}

        {(searchResults || isLoading) && (
          <ResultsList results={searchResults} isLoading={isLoading} />
        )}

        {!searchResults && !isLoading && !error && (
          <div className="text-center py-16 bg-white rounded-lg border border-slate-200 shadow-sm">
            <h2 className="text-xl font-semibold mb-3 text-slate-800">
              Enter a search query to get started
            </h2>
            <p className="text-slate-600 mb-6 max-w-2xl mx-auto">
              Describe your patient's situation or the type of guidance you're
              looking for. Our system will retrieve relevant conversations and identify patterns.
            </p>
            {labels && (
              <div className="max-w-2xl mx-auto px-6">
                <p className="text-sm font-medium text-slate-700 mb-3">
                  Common issues in our database:
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {labels.issues.slice(0, 10).map((issue, index) => (
                    <span
                      key={index}
                      className="bg-slate-100 text-slate-700 border border-slate-200 text-xs px-3 py-1 rounded-full"
                    >
                      {issue}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;