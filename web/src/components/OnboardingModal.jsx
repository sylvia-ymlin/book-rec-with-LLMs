/**
 * P2: New-user onboarding — pick 3–5 books to seed preferences.
 * Shown when myCollection is empty and onboarding not completed.
 */
import React, { useState, useEffect } from "react";
import { getOnboardingBooks } from "../api";

const PLACEHOLDER_IMG = "/content/cover-not-found.jpg";
const MIN_SELECT = 3;
const MAX_SELECT = 5;

const OnboardingModal = ({ onComplete, onAddFavorite, onSkip }) => {
  const [books, setBooks] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getOnboardingBooks(24)
      .then(setBooks)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (isbn) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(isbn)) {
        next.delete(isbn);
      } else if (next.size < MAX_SELECT) {
        next.add(isbn);
      }
      return next;
    });
  };

  const handleComplete = async () => {
    if (selected.size < MIN_SELECT) return;
    try {
      for (const isbn of selected) {
        await onAddFavorite(isbn);
      }
      localStorage.setItem("onboarding_complete", "true");
      onComplete();
    } catch (e) {
      setError(e.message);
    }
  };

  const canComplete = selected.size >= MIN_SELECT;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white max-w-3xl w-full max-h-[90vh] overflow-hidden shadow-xl">
        <div className="p-6 border-b border-[#eee]">
          <h2 className="text-xl font-bold text-[#333]">Welcome — Pick Your Favorites</h2>
          <p className="text-sm text-gray-500 mt-1">
            Select 3–5 books you like to get personalized recommendations.
          </p>
        </div>
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {loading && (
            <div className="text-center text-gray-400 py-8">Loading popular books...</div>
          )}
          {error && (
            <div className="text-center text-red-500 py-4 text-sm">{error}</div>
          )}
          {!loading && !error && (
            <div className="grid grid-cols-3 md:grid-cols-4 gap-4">
              {books.map((book) => {
                const isSelected = selected.has(book.isbn);
                return (
                  <button
                    key={book.isbn}
                    type="button"
                    onClick={() => toggle(book.isbn)}
                    className={`text-left border-2 transition-all p-2 ${
                      isSelected ? "border-[#b392ac] bg-[#faf5f7]" : "border-[#eee] hover:border-[#ddd]"
                    }`}
                  >
                    <div className="aspect-[3/4] bg-gray-100 mb-2 overflow-hidden">
                      <img
                        src={book.thumbnail || PLACEHOLDER_IMG}
                        alt={book.title}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = PLACEHOLDER_IMG;
                        }}
                      />
                    </div>
                    <p className="text-[10px] font-bold text-[#555] truncate" title={book.title}>
                      {book.title}
                    </p>
                    {isSelected && (
                      <span className="text-[10px] text-[#b392ac] font-bold">✓ Selected</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
        <div className="p-6 border-t border-[#eee] flex justify-between items-center">
          <span className="text-xs text-gray-500">
            {selected.size} selected (min {MIN_SELECT}, max {MAX_SELECT})
          </span>
          <div className="flex gap-2">
            {onSkip && (
              <button
                type="button"
                onClick={() => {
                  localStorage.setItem("onboarding_complete", "true");
                  onSkip();
                }}
                className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700"
              >
                Skip for now
              </button>
            )}
          <button
            onClick={handleComplete}
            disabled={!canComplete}
            className={`px-6 py-2 text-sm font-bold ${
              canComplete ? "bg-[#b392ac] text-white" : "bg-gray-200 text-gray-400 cursor-not-allowed"
            }`}
          >
            Start Exploring
          </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingModal;
