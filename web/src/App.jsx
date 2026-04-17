import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import {
  recommend,
  addFavorite,
  getHighlights,
  streamChat,
  getFavorites,
  getUserStats,
  addBook,
  searchGoogleBooks,
  getPersonalizedRecommendations,
} from "./api";
import { mapRecommendationsToCards } from "./utils/recommendationMapper";
import { useSessionRecents } from "./hooks/useSessionRecents";
import { useLlmSettings } from "./hooks/useLlmSettings";
import { useCollectionActions } from "./hooks/useCollectionActions";

// Components
import Header from "./components/Header";
import BookDetailModal from "./components/BookDetailModal";
import SettingsModal from "./components/SettingsModal";
import AddBookModal from "./components/AddBookModal";
import OnboardingModal from "./components/OnboardingModal";

// Pages
import GalleryPage from "./pages/GalleryPage";
import BookshelfPage from "./pages/BookshelfPage";
import ProfilePage from "./pages/ProfilePage";

const App = () => {
  // --- Core State ---
  const [userId, setUserId] = useState("Coconut");
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
  const { apiKey, setApiKey, llmProvider, setLlmProvider, saveSettings } =
    useLlmSettings();

  // --- P1: Session-level recent ISBNs for cold-start ---
  const { recentIsbns, trackRecentIsbn } = useSessionRecents();

  // --- P2: Onboarding (new user, no collection) ---
  const [showOnboarding, setShowOnboarding] = useState(false);

  // --- Add Book Modal State ---
  const [showAddBook, setShowAddBook] = useState(false);
  const [googleQuery, setGoogleQuery] = useState("");
  const [googleResults, setGoogleResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [addingBookId, setAddingBookId] = useState(null);

  const {
    refreshCollection,
    toggleCollect,
    handleRatingChange,
    handleStatusChange,
    handleRemoveBook,
    handleUpdateComment,
  } = useCollectionActions({
    userId,
    myCollection,
    setMyCollection,
    setReadingStats,
  });

  // --- P2: Show onboarding when new user (no collection, not completed) ---
  useEffect(() => {
    const completed = localStorage.getItem("onboarding_complete") === "true";
    if (!completed && userId === "Coconut") {
      setShowOnboarding(true);
    }
  }, [userId]);

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
      getPersonalizedRecommendations(userId, 20, recentIsbns).catch(() => []),
    ]).then(([favs, stats, personalRecs]) => {
      setMyCollection(favs);
      setReadingStats(stats);
      if (favs.length > 0) {
        localStorage.setItem("onboarding_complete", "true");
      }
      setBooks(mapRecommendationsToCards(personalRecs));
      setLoading(false);
    });
  }, [userId]);

  // --- Handlers ---
  const handleSaveSettings = () => {
    saveSettings();
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
      userId,
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
      await refreshCollection();
    } catch (err) {
      alert("Import failed: " + err.message);
    } finally {
      setAddingBookId(null);
    }
  };

  const openBook = (book) => {
    // P1: Track session-level recent views for cold-start
    if (book?.isbn) {
      trackRecentIsbn(book.isbn);
    }
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
      // P2: Cold-start with intent — when no collection and user typed a mood, use intent-seeded personal recs
      const useIntentSeed = myCollection.length === 0 && searchQuery.trim();
      if (!searchQuery || useIntentSeed) {
        recs = await getPersonalizedRecommendations(
          userId,
          20,
          recentIsbns,
          useIntentSeed ? searchQuery : null
        );
      } else {
        recs = await recommend(searchQuery, searchCategory, searchMood, userId);
      }
      setBooks(
        mapRecommendationsToCards(recs, {
          category: searchCategory,
          mood: searchMood,
          suggestedQuestions: [
            "Matches my current mood?",
            "Any similar recommendations?",
            "What's the core highlight?",
          ],
        })
      );
    } catch (err) {
      setError(err.message || "Failed to get recommendations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <BrowserRouter>
      <div className="min-h-screen text-text-primary font-sans tracking-tight bg-background">
        <div>
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
            onSave={handleSaveSettings}
          />
        )}

        {showOnboarding && (
          <OnboardingModal
            onComplete={async (selectedIsbns) => {
              setShowOnboarding(false);
              // Track in session for subsequent calls
              selectedIsbns.forEach(trackRecentIsbn);
              
              const [favs, stats, personalRecs] = await Promise.all([
                getFavorites(userId).catch(() => []),
                getUserStats(userId).catch(() => ({ total: 0, want_to_read: 0, reading: 0, finished: 0 })),
                getPersonalizedRecommendations(userId, 20, selectedIsbns).catch(() => []), 
              ]);
              setMyCollection(favs);
              setReadingStats(stats);
              setBooks(mapRecommendationsToCards(personalRecs));
            }}
            onAddFavorite={(isbn) => addFavorite(isbn, userId)}
            onSkip={() => setShowOnboarding(false)}
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

          <footer className="mt-16 text-center text-[9px] font-medium text-[#81A1C1] uppercase tracking-widest pb-10 border-t border-[#88C0D0]/80 pt-10">
            Book Shelf // 2026 Your Personal Library
          </footer>
        </div>
      </div>
    </BrowserRouter>
  );
};

export default App;
