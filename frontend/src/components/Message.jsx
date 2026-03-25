import React from "react";

export default function Message({ msg }) {
  return (
    <div className={`message ${msg.role}`}>
      <p>{msg.text}</p>

      {msg.sources && (
        <div className="sources">
          <strong>Sources:</strong>
          {msg.sources.map((s, i) => (
            <div key={i}>
              {s.source} (page {s.page})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}