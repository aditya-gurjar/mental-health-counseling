// src/components/LabelDisplay.jsx
import React from "react";

const LabelDisplay = ({ predictions }) => {
  if (!predictions) return null;

  const renderLabelGroup = (labels, title, bgColor, textColor, borderColor) => {
    return (
      <div className="mb-5">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">{title}</h3>
        <div className="flex flex-wrap gap-2">
          {labels.map(({ label, confidence }, index) => (
            <span
              key={index}
              className={`${bgColor} ${textColor} ${borderColor} text-xs font-medium px-3 py-1 rounded-full flex items-center border`}
            >
              {label}
              <span className="ml-1 opacity-75">
                {Math.round(confidence * 100)}%
              </span>
            </span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 mb-6">
      <h2 className="text-lg font-semibold mb-4 text-slate-800">Detected Patterns</h2>
      {predictions.issues && predictions.issues.length > 0 && 
        renderLabelGroup(
          predictions.issues,
          "Patient Issues",
          "bg-red-50",
          "text-red-700",
          "border-red-200"
        )}
      {predictions.response_types && predictions.response_types.length > 0 && 
        renderLabelGroup(
          predictions.response_types,
          "Response Types",
          "bg-blue-50",
          "text-blue-700",
          "border-blue-200"
        )}
      {predictions.approaches && predictions.approaches.length > 0 && 
        renderLabelGroup(
          predictions.approaches,
          "Therapeutic Approaches",
          "bg-green-50",
          "text-green-700",
          "border-green-200"
        )}
    </div>
  );
};

export default LabelDisplay;