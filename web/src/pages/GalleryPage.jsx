import React from "react";
import { Search, Layers, Smile } from "lucide-react";
import BookCard from "../components/BookCard";
import { SEARCH_CATEGORIES, SEARCH_MOODS } from "../constants";

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
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center bg-white border border-[#cddabf] rounded-[32px] p-4 shadow-soft">
          <div className="md:col-span-6 flex items-center bg-[#F7F9F2] border border-[#d9e4cf] p-3 rounded-full">
            <Search className="w-4 h-4 mr-3 text-[#81A1C1] ml-2" />
            <input
              className="w-full outline-none text-sm text-text-primary placeholder-text-secondary bg-transparent font-sans"
              placeholder="Search for a topic, mood, or dream..."
              value={searchQuery}
              onChange={(e) => onSearchQueryChange(e.target.value)}
            />
          </div>
          <div className="md:col-span-3 flex items-center bg-[#F7F9F2] border border-[#d9e4cf] p-3 rounded-full">
            <Layers className="w-4 h-4 mr-3 text-[#81A1C1] ml-2" />
            <select
              className="w-full outline-none text-sm bg-transparent text-text-primary font-sans"
              value={searchCategory}
              onChange={(e) => onSearchCategoryChange(e.target.value)}
            >
              {SEARCH_CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          <div className="md:col-span-3 flex items-center bg-[#F7F9F2] border border-[#d9e4cf] p-3 rounded-full">
            <Smile className="w-4 h-4 mr-3 text-[#81A1C1] ml-2" />
            <select
              className="w-full outline-none text-sm bg-transparent text-text-primary font-sans"
              value={searchMood}
              onChange={(e) => onSearchMoodChange(e.target.value)}
            >
              {SEARCH_MOODS.map((mood) => (
                <option key={mood} value={mood}>{mood}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex justify-center">
          <button
            onClick={onStartDiscovery}
            className="px-12 py-3 text-sm font-semibold transition-all bg-text-primary text-surface hover:opacity-90 rounded-full shadow-soft"
          >
            Start Discovery
          </button>
        </div>
        {loading && <div className="text-center text-xs text-gray-400">Loading...</div>}
        {error && <div className="text-center text-xs text-[#BF616A]">{error}</div>}
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
