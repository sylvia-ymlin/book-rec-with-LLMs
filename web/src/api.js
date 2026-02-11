const API_URL = import.meta.env.VITE_API_URL || "http://localhost:6006";

export async function recommend(query, category = "All", tone = "All") {
  const body = { query, category, tone };
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

export async function getHighlights(isbn, userId = "local") {
  const resp = await fetch(`${API_URL}/marketing/highlights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ isbn, user_id: userId }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}
