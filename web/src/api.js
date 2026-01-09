const API_URL = import.meta.env.VITE_API_URL || "http://localhost:6006";

export async function recommend(query, category = "All", tone = "All", user_id = "local") {
  const body = { query, category, tone, user_id };
  const resp = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.recommendations || [];
}

export async function addFavorite(isbn, userId = "local") {
  const resp = await fetch(`${API_URL}/favorites/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getPersona(userId = "local") {
  const resp = await fetch(`${API_URL}/user/${userId}/persona`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getFavorites(userId = "local") {
  const resp = await fetch(`${API_URL}/favorites/list/${userId}`);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.favorites || [];
}

export async function updateBook(isbn, updates, userId = "local") {
  const resp = await fetch(`${API_URL}/favorites/update`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId, ...updates }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function removeFromFavorites(isbn, userId = "local") {
  const resp = await fetch(`${API_URL}/favorites/remove`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getUserStats(userId = "local") {
  const resp = await fetch(`${API_URL}/user/${userId}/stats`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getHighlights(isbn, userId = "local") {
  const resp = await fetch(`${API_URL}/marketing/highlights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}
export async function streamChat({ isbn, query, apiKey, provider, onChunk, onError }) {
  try {
    const resp = await fetch(`${API_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-LLM-Key": apiKey || ""
      },
      body: JSON.stringify({
        isbn,
        query,
        user_id: "local",
        provider: provider || (apiKey ? "openai" : "mock")
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(errText || resp.statusText);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      if (chunk) onChunk(chunk);
    }
  } catch (e) {
    if (onError) onError(e);
    else console.error(e);
  }
}
