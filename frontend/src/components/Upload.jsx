import React, { useState } from "react";
import { uploadFile } from "../api";

export default function Upload({ sessionId }) {
  const [status, setStatus] = useState("");

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setStatus("Uploading...");

    await uploadFile(file, sessionId);

    setStatus("Indexed successfully");
  };

  return (
    <div className="upload">
      <h3>Upload Document</h3>
      <input type="file" onChange={handleUpload} />
      <p>{status}</p>
    </div>
  );
}