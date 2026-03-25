import React, { useEffect, useState } from "react";
import { createSession } from "./api";
import Upload from "./components/Upload";
import Chat from "./components/Chat";

export default function App() {
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    async function init() {
      const data = await createSession();
      setSessionId(data.session_id);
    }
    init();
  }, []);

  if (!sessionId) return <div>Loading session...</div>;

  return (
    <div className="container">
      <h1>Financial RAG Assistant</h1>
      <Upload sessionId={sessionId} />
      <Chat sessionId={sessionId} />
    </div>
  );
}