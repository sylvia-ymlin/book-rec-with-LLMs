const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "" : "http://127.0.0.1:6006");

export async function recommend(query, category = "All", tone = "All", user_id = "Coconut", use_agentic = false, fast = false, async_rerank = false) {
  const body = { query, category, tone, user_id, use_agentic, fast, async_rerank };
  const resp = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.recommendations || [];
}

export async function getOnboardingBooks(limit = 24) {
  const resp = await fetch(`${API_URL}/api/onboarding/books?limit=${limit}`);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.books || [];
}

export async function getPersonalizedRecommendations(user_id = "Coconut", limit = 20, recent_isbns = null, intent_query = null) {
  // P1: recent_isbns — session-level ISBNs for cold-start (1+ clicks)
  // P2: intent_query — zero-shot intent probing when user has no history
  const params = new URLSearchParams({ user_id, limit: limit.toString() });
  if (recent_isbns && Array.isArray(recent_isbns) && recent_isbns.length > 0) {
    params.set("recent_isbns", recent_isbns.join(","));
  }
  if (intent_query && typeof intent_query === "string" && intent_query.trim()) {
    params.set("intent_query", intent_query.trim());
  }
  const resp = await fetch(`${API_URL}/api/recommend/personal?${params.toString()}`);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.recommendations || [];
}

export async function getSimilarBooks(isbn, k = 6, category = "All") {
  const params = new URLSearchParams({ k: k.toString(), category });
  const resp = await fetch(`${API_URL}/api/recommend/similar/${encodeURIComponent(isbn)}?${params.toString()}`);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.recommendations || [];
}

export async function addFavorite(isbn, userId = "Coconut") {
  const resp = await fetch(`${API_URL}/favorites/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getPersona(userId = "Coconut") {
  const resp = await fetch(`${API_URL}/user/${userId}/persona`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getFavorites(userId = "Coconut") {
  const resp = await fetch(`${API_URL}/favorites/list/${userId}`);
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data.favorites || [];
}

export async function updateBook(isbn, updates, userId = "Coconut") {
  const resp = await fetch(`${API_URL}/favorites/update`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId, ...updates }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function removeFromFavorites(isbn, userId = "Coconut") {
  const resp = await fetch(`${API_URL}/favorites/remove`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getUserStats(userId = "Coconut") {
  const resp = await fetch(`${API_URL}/user/${userId}/stats`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getHighlights(isbn, userId = "Coconut") {
  const resp = await fetch(`${API_URL}/marketing/highlights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}
export async function streamChat({ isbn, query, userId = "Coconut", apiKey, provider, onChunk, onError }) {
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
        user_id: userId,
        provider: provider || "ollama"
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

export async function addBook(bookData) {
  const resp = await fetch(`${API_URL}/books/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bookData),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function searchGoogleBooks(query) {
  const resp = await fetch(`https://www.googleapis.com/books/v1/volumes?q=${encodeURIComponent(query)}&maxResults=5`);
  if (!resp.ok) throw new Error("Failed to search Google Books");
  const data = await resp.json();
  return data.items || [];
}
