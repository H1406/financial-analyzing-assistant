import React, { useState } from "react";
import { queryRAG } from "../api";
import Message from "./Message";

export default function Chat({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = async () => {
    if (!input) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);

    setInput("");

    const res = await queryRAG(sessionId, input);

    const botMessage = {
      role: "assistant",
      text: res.answer,
      sources: res.sources,
    };

    setMessages((prev) => [...prev, botMessage]);
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
      </div>

      <div className="input-box">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}