function getDominantMood(emotions) {
  if (!emotions || Object.keys(emotions).length === 0) {
    return "Literary";
  }
  return Object.entries(emotions).reduce((a, b) => (a[1] > b[1] ? a : b))[0];
}

export function mapRecommendationToCard(rec, idx, options = {}) {
  const category = options.category ?? rec.category ?? "General";
  const mood =
    options.mood && options.mood !== "All"
      ? options.mood
      : getDominantMood(rec.emotions);

  return {
    id: rec.isbn,
    title: rec.title,
    author: rec.authors,
    category,
    mood,
    rank: idx + 1,
    rating: rec.average_rating || 0,
    tags: rec.tags || [],
    review_highlights: rec.review_highlights || [],
    desc: rec.description,
    img: rec.thumbnail,
    isbn: rec.isbn,
    emotions: rec.emotions || {},
    explanations: rec.explanations || [],
    aiHighlight: "\u2014",
    suggestedQuestions: options.suggestedQuestions || [
      "Why was this recommended?",
      "Similar to what I've read?",
      "What's the core highlight?",
    ],
  };
}

export function mapRecommendationsToCards(recommendations, options = {}) {
  return (recommendations || []).map((rec, idx) =>
    mapRecommendationToCard(rec, idx, options)
  );
}

