const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers }, ...options,
  });
  if (!response.ok) {
    let message = "通信に失敗しました";
    try { message = (await response.json()).detail || message; } catch { /* no JSON body */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  chats: () => request("/chats"),
  createChat: () => request("/chats", { method: "POST", body: "{}" }),
  renameChat: (id, title) => request(`/chats/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteChat: (id) => request(`/chats/${id}`, { method: "DELETE" }),
  messages: (id) => request(`/chats/${id}/messages`),
  send: (id, content) => request(`/chats/${id}/messages`, { method: "POST", body: JSON.stringify({ content }) }),
  sendTemporary: (content, history) => request("/temporary/messages", { method: "POST", body: JSON.stringify({ content, history: history.slice(-10) }) }),
};
