import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import {
  recommend,
  addFavorite,
  getHighlights,
  streamChat,
  getFavorites,
  updateBook,
  removeFromFavorites,
  getUserStats,
  addBook,
  searchGoogleBooks,
  getPersonalizedRecommendations,
} from "./api";

// Components
import Header from "./components/Header";
import BookDetailModal from "./components/BookDetailModal";
import SettingsModal from "./components/SettingsModal";
import AddBookModal from "./components/AddBookModal";

// Pages
import GalleryPage from "./pages/GalleryPage";
import BookshelfPage from "./pages/BookshelfPage";
import ProfilePage from "./pages/ProfilePage";

const App = () => {
  // --- Core State ---
  const [userId, setUserId] = useState("local");
  const [myCollection, setMyCollection] = useState([]);
  const [readingStats, setReadingStats] = useState({
    total: 0,
    want_to_read: 0,
    reading: 0,
    finished: 0,
  });
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // --- Book Detail Modal State ---
  const [selectedBook, setSelectedBook] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  // --- Search State ---
  const [searchQuery, setSearchQuery] = useState("");
  const [searchCategory, setSearchCategory] = useState("All");
  const [searchMood, setSearchMood] = useState("All");

  // --- Settings State ---
  const [showSettings, setShowSettings] = useState(false);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("openai_key") || "");
  const [llmProvider, setLlmProvider] = useState(() => {
    const stored = localStorage.getItem("llm_provider");
    return stored === "mock" || !stored ? "ollama" : stored;
  });

  // --- Add Book Modal State ---
  const [showAddBook, setShowAddBook] = useState(false);
  const [googleQuery, setGoogleQuery] = useState("");
  const [googleResults, setGoogleResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [addingBookId, setAddingBookId] = useState(null);

  // --- Load favorites and stats on startup or user change ---
  useEffect(() => {
    setLoading(true);
    setMyCollection([]);
    setMessages([]);

    Promise.all([
      getFavorites(userId).catch(() => []),
      getUserStats(userId).catch(() => ({
        total: 0,
        want_to_read: 0,
        reading: 0,
        finished: 0,
      })),
      getPersonalizedRecommendations(userId).catch(() => []),
    ]).then(([favs, stats, personalRecs]) => {
      setMyCollection(favs);
      setReadingStats(stats);

      const mappedRecs = personalRecs.map((r, idx) => ({
        id: r.isbn,
        title: r.title,
        author: r.authors,
        category: r.category || "General",
        mood:
          r.emotions && Object.keys(r.emotions).length > 0
            ? Object.entries(r.emotions).reduce((a, b) =>
                a[1] > b[1] ? a : b
              )[0]
            : "Literary",
        rank: idx + 1,
        rating: r.average_rating || 0,
        tags: r.tags || [],
        review_highlights: r.review_highlights || [],
        desc: r.description,
        img: r.thumbnail,
        isbn: r.isbn,
        emotions: r.emotions || {},
        explanations: r.explanations || [],
        aiHighlight: "\u2014",
        suggestedQuestions: [
          "Why was this recommended?",
          "Similar to what I've read?",
          "What's the core highlight?",
        ],
      }));

      setBooks(mappedRecs);
      setLoading(false);
    });
  }, [userId]);

  // --- Handlers ---
  const saveSettings = () => {
    localStorage.setItem("openai_key", apiKey);
    localStorage.setItem("llm_provider", llmProvider);
    setShowSettings(false);
  };

  const handleSend = async (text) => {
    if (!text) return;
    const newMsgs = [...messages, { role: "user", content: text }];
    setMessages(newMsgs);
    setInput("");

    setMessages((prev) => [...prev, { role: "ai", content: "Thinking..." }]);
    const aiMsgIndex = newMsgs.length;

    let currentAiMsg = "";
    await streamChat({
      isbn: selectedBook.isbn,
      query: text,
      apiKey: apiKey,
      provider: llmProvider,
      onChunk: (chunk) => {
        currentAiMsg += chunk;
        setMessages((prev) => {
          const updated = [...prev];
          updated[aiMsgIndex] = { role: "ai", content: currentAiMsg };
          return updated;
        });
      },
      onError: (err) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[aiMsgIndex] = {
            role: "ai",
            content: `Error: ${err.message}. Check your API Key in Settings.`,
          };
          return updated;
        });
      },
    });
  };

  const handleSearchGoogle = async (e) => {
    e.preventDefault();
    if (!googleQuery.trim()) return;
    setIsSearching(true);
    setGoogleResults([]);
    try {
      const items = await searchGoogleBooks(googleQuery);
      setGoogleResults(items);
    } catch (err) {
      console.error(err);
      alert("Search failed: " + err.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleImportBook = async (item) => {
    setAddingBookId(item.id);
    const info = item.volumeInfo;
    let isbn = item.id;
    if (info.industryIdentifiers) {
      const isbn13 = info.industryIdentifiers.find((i) => i.type === "ISBN_13");
      const isbn10 = info.industryIdentifiers.find((i) => i.type === "ISBN_10");
      isbn = isbn13 ? isbn13.identifier : isbn10 ? isbn10.identifier : item.id;
    }

    const bookData = {
      isbn,
      title: info.title || "Unknown Title",
      author: info.authors ? info.authors.join(", ") : "Unknown Author",
      description: info.description || "No description provided.",
      category: info.categories ? info.categories[0] : "General",
      thumbnail: info.imageLinks?.thumbnail || info.imageLinks?.smallThumbnail || null,
    };

    try {
      await addBook(bookData);
      await addFavorite(bookData.isbn, userId);
      alert(`Successfully imported "${bookData.title}" to your collection!`);
      setShowAddBook(false);
      setGoogleResults([]);
      setGoogleQuery("");

      const [favs, stats] = await Promise.all([
        getFavorites(userId),
        getUserStats(userId),
      ]);
      setMyCollection(favs);
      setReadingStats(stats);
    } catch (err) {
      alert("Import failed: " + err.message);
    } finally {
      setAddingBookId(null);
    }
  };

  const toggleCollect = async (book) => {
    try {
      if (myCollection.some((b) => b.isbn === book.isbn)) {
        await removeFromFavorites(book.isbn, userId);
      } else {
        await addFavorite(book.isbn, userId);
      }
      const [favs, stats] = await Promise.all([
        getFavorites(userId),
        getUserStats(userId),
      ]);
      setMyCollection(favs);
      setReadingStats(stats);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRatingChange = async (isbn, rating) => {
    try {
      await updateBook(isbn, { rating }, userId);
      setMyCollection((prev) =>
        prev.map((book) => (book.isbn === isbn ? { ...book, rating } : book))
      );
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleStatusChange = async (isbn, status) => {
    try {
      await updateBook(isbn, { status }, userId);
      setMyCollection((prev) =>
        prev.map((book) => (book.isbn === isbn ? { ...book, status } : book))
      );
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRemoveBook = async (isbn) => {
    try {
      await removeFromFavorites(isbn, userId);
      setMyCollection((prev) => prev.filter((book) => book.isbn !== isbn));
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpdateComment = (isbn, value, persist) => {
    setMyCollection((prev) =>
      prev.map((b) => (b.isbn === isbn ? { ...b, comment: value } : b))
    );
    if (persist) {
      updateBook(isbn, { comment: value }, userId).catch(console.error);
    }
  };

  const openBook = (book) => {
    setSelectedBook({
      ...book,
      aiHighlight: "\u2728 ...",
      suggestedQuestions: [
        "Who is the target audience for this book?",
        "Does the author have similar works?",
        "Can you summarize the main content?",
      ],
    });
    setMessages([]);

    getHighlights(book.isbn)
      .then((res) => {
        const meta = res?.meta || {};
        const rawHighlight = (res?.highlights || []).join("\n") || "\u2014";
        const cleanHighlight = rawHighlight.replace(/^["']|["']$/g, "").trim();
        setSelectedBook((prev) => ({
          ...prev,
          aiHighlight: cleanHighlight,
          desc: meta?.description || prev.desc,
        }));
      })
      .catch(() => {
        setSelectedBook((prev) => ({
          ...prev,
          aiHighlight: "Unable to generate highlight.",
        }));
      });
  };

  const startDiscovery = async () => {
    setLoading(true);
    setError("");
    setBooks([]);
    try {
      let recs;
      if (!searchQuery) {
        recs = await getPersonalizedRecommendations(userId);
      } else {
        recs = await recommend(searchQuery, searchCategory, searchMood, userId);
      }
      const mapped = (recs || []).map((r, idx) => ({
        id: r.isbn,
        title: r.title,
        author: r.authors,
        category: searchCategory,
        mood:
          searchMood !== "All"
            ? searchMood
            : r.emotions && Object.keys(r.emotions).length > 0
            ? Object.entries(r.emotions).reduce((a, b) =>
                a[1] > b[1] ? a : b
              )[0]
            : "Literary",
        rank: idx + 1,
        rating: r.average_rating || 0,
        tags: r.tags || [],
        review_highlights: r.review_highlights || [],
        desc: r.description,
        img: r.thumbnail,
        isbn: r.isbn,
        emotions: r.emotions || {},
        explanations: r.explanations || [],
        aiHighlight: "\u2014",
        suggestedQuestions: [
          "Matches my current mood?",
          "Any similar recommendations?",
          "What's the core highlight?",
        ],
      }));
      setBooks(mapped);
    } catch (err) {
      setError(err.message || "Failed to get recommendations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#faf9f6] text-[#444] font-serif tracking-tight">
        {/* Shared Header */}
        <Header
          userId={userId}
          onUserIdChange={setUserId}
          onAddBookClick={() => setShowAddBook(true)}
          onSettingsClick={() => setShowSettings(true)}
        />

        {/* Global Modals */}
        {showSettings && (
          <SettingsModal
            onClose={() => setShowSettings(false)}
            apiKey={apiKey}
            onApiKeyChange={setApiKey}
            llmProvider={llmProvider}
            onProviderChange={setLlmProvider}
            onSave={saveSettings}
          />
        )}

        {showAddBook && (
          <AddBookModal
            onClose={() => setShowAddBook(false)}
            googleQuery={googleQuery}
            onQueryChange={setGoogleQuery}
            googleResults={googleResults}
            isSearching={isSearching}
            addingBookId={addingBookId}
            onSearch={handleSearchGoogle}
            onImport={handleImportBook}
          />
        )}

        {selectedBook && (
          <BookDetailModal
            book={selectedBook}
            onClose={() => setSelectedBook(null)}
            messages={messages}
            onSend={handleSend}
            input={input}
            onInputChange={setInput}
            myCollection={myCollection}
            onToggleCollect={toggleCollect}
            onRatingChange={handleRatingChange}
            onStatusChange={handleStatusChange}
            onUpdateComment={handleUpdateComment}
            onOpenBook={openBook}
          />
        )}

        {/* Route Pages */}
        <main className="max-w-5xl mx-auto px-4 pb-20">
          <Routes>
            <Route
              path="/"
              element={
                <GalleryPage
                  books={books}
                  loading={loading}
                  error={error}
                  searchQuery={searchQuery}
                  onSearchQueryChange={setSearchQuery}
                  searchCategory={searchCategory}
                  onSearchCategoryChange={setSearchCategory}
                  searchMood={searchMood}
                  onSearchMoodChange={setSearchMood}
                  onStartDiscovery={startDiscovery}
                  myCollection={myCollection}
                  onOpenBook={openBook}
                />
              }
            />
            <Route
              path="/bookshelf"
              element={
                <BookshelfPage
                  myCollection={myCollection}
                  readingStats={readingStats}
                  onOpenBook={openBook}
                  onRemoveBook={handleRemoveBook}
                  onRatingChange={handleRatingChange}
                  onStatusChange={handleStatusChange}
                />
              }
            />
            <Route
              path="/profile"
              element={
                <ProfilePage
                  userId={userId}
                  myCollection={myCollection}
                  readingStats={readingStats}
                />
              }
            />
          </Routes>
        </main>

        <footer className="mt-16 text-center text-[9px] font-medium text-gray-300 uppercase tracking-widest pb-10 border-t border-[#eee] pt-10">
          Paper Shelf // 2026 Your Personal Library
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
