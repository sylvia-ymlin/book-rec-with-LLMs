/**
 * P2: New-user onboarding — pick 3–5 books to seed preferences.
 * Shown when myCollection is empty and onboarding not completed.
 */
import React, { useState, useEffect } from "react";
import { getOnboardingBooks } from "../api";
import { PLACEHOLDER_IMG } from "../constants";
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
        <div className="p-6 border-b border-[#88C0D0]">
          <h2 className="text-xl font-bold text-[#4C566A]">Welcome - Pick Your Favorites</h2>
          <p className="text-sm text-[#5E81AC] mt-1">
            Select 3–5 books you like to get personalized recommendations.
          </p>
        </div>
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {loading && (
            <div className="text-center text-gray-400 py-8">Loading popular books...</div>
          )}
          {error && (
            <div className="text-center text-[#BF616A] py-4 text-sm">{error}</div>
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
                      isSelected ? "border-[#5E81AC] bg-[#D8DEE9]/35" : "border-[#88C0D0] hover:border-[#81A1C1]"
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
                    <p className="text-[10px] font-bold text-[#4C566A] truncate" title={book.title}>
                      {book.title}
                    </p>
                    {isSelected && (
                      <span className="text-[10px] text-[#5E81AC] font-bold">Selected</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
        <div className="p-6 border-t border-[#88C0D0] flex justify-between items-center">
          <span className="text-xs text-[#4C566A]">
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
                className="px-4 py-2 text-sm text-[#5E81AC] hover:text-[#4C566A]"
              >
                Skip for now
              </button>
            )}
          <button
            onClick={handleComplete}
            disabled={!canComplete}
            className={`px-6 py-2 text-sm font-bold ${
              canComplete ? "bg-[#5E81AC] text-white" : "bg-[#D8DEE9] text-[#81A1C1] cursor-not-allowed"
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
