const BASE_URL = "http://localhost:8000";

export async function createSession() {
  const res = await fetch(`${BASE_URL}/session`);
  return await res.json();
}

export async function uploadFile(file, sessionId) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload?session_id=${sessionId}`, {
    method: "POST",
    body: formData,
  });

  return await res.json();
}

export async function queryRAG(sessionId, query) {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      query: query,
    }),
  });

  return await res.json();
}