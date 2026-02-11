import React from "react";
import { Search, Layers, Smile } from "lucide-react";
import BookCard from "../components/BookCard";

const CATEGORIES = ["All", "Fiction", "History", "Philosophy", "Science", "Art"];
const MOODS = ["All", "Happy", "Suspenseful", "Angry", "Sad", "Surprising"];

const GalleryPage = ({
  books,
  loading,
  error,
  searchQuery,
  onSearchQueryChange,
  searchCategory,
  onSearchCategoryChange,
  searchMood,
  onSearchMoodChange,
  onStartDiscovery,
  myCollection,
  onOpenBook,
}) => {
  return (
    <>
      {/* Search Bar */}
      <div className="max-w-4xl mx-auto mb-16 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
          <div className="md:col-span-6 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
            <Search className="w-4 h-4 mr-3 text-gray-300 ml-2" />
            <input
              className="w-full outline-none text-sm placeholder-gray-400 bg-transparent font-serif"
              placeholder="Search for a topic, mood, or dream..."
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
            />
          </div>
          <div className="md:col-span-3 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
            <Layers className="w-4 h-4 mr-3 text-gray-300 ml-2" />
            <select
              className="w-full outline-none text-sm bg-transparent text-gray-500 font-serif"
              value={searchCategory}
              onChange={(e) => onSearchCategoryChange(e.target.value)}
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          <div className="md:col-span-3 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
            <Smile className="w-4 h-4 mr-3 text-gray-300 ml-2" />
            <select
              className="w-full outline-none text-sm bg-transparent text-gray-500 font-serif"
              value={searchMood}
              onChange={(e) => onSearchMoodChange(e.target.value)}
            >
              {MOODS.map((mood) => (
                <option key={mood} value={mood}>{mood}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex justify-center">
          <button
            onClick={onStartDiscovery}
            className="px-12 py-2 text-sm font-bold transition-all bg-[#b392ac] text-white hover:bg-[#9d7799]"
          >
            Start Discovery
          </button>
        </div>
        {loading && <div className="text-center text-xs text-gray-400">Loading...</div>}
        {error && <div className="text-center text-xs text-red-400">{error}</div>}
      </div>

      {/* Book Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
        {books.length > 0 ? (
          books.map((book, idx) => (
            <BookCard
              key={book.isbn || idx}
              book={book}
              showShelfControls={false}
              isInCollection={myCollection.some((b) => b.isbn === book.isbn)}
              onOpenBook={onOpenBook}
            />
          ))
        ) : (
          !loading && (
            <div className="col-span-full py-20 text-center text-gray-400 text-xs italic">
              No books here yet. Start discovering to build your collection.
            </div>
          )
        )}
      </div>
    </>
  );
};

export default GalleryPage;
