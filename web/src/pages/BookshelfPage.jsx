import React, { useState } from "react";
import { BarChart3 } from "lucide-react";
import BookCard from "../components/BookCard";

const BookshelfPage = ({
  myCollection,
  readingStats,
  onOpenBook,
  onRemoveBook,
  onRatingChange,
  onStatusChange,
}) => {
  const [shelfFilter, setShelfFilter] = useState("all");
  const [shelfSort, setShelfSort] = useState("recent");

  const getFilteredShelf = () => {
    let filtered = [...myCollection];

    // Filter
    if (shelfFilter !== "all") {
      filtered = filtered.filter((b) => b.status === shelfFilter);
    }

    // Sort
    if (shelfSort === "rating_high") {
      filtered.sort((a, b) => (b.rating || 0) - (a.rating || 0));
    } else if (shelfSort === "rating_low") {
      filtered.sort((a, b) => (a.rating || 0) - (b.rating || 0));
    } else if (shelfSort === "title") {
      filtered.sort((a, b) => a.title.localeCompare(b.title));
    } else {
      // Recent (default) - reverse for newest first
      filtered.reverse();
    }

    return filtered;
  };

  const filteredBooks = getFilteredShelf();

  return (
    <>
      <div className="mb-8 space-y-4">
        {/* Shelf Controls */}
        <div className="flex justify-between items-center bg-white p-3 border border-[#eee] shadow-sm mb-4">
          <div className="flex gap-2">
            {["all", "want_to_read", "reading", "finished"].map((status) => (
              <button
                key={status}
                onClick={() => setShelfFilter(status)}
                className={`px-3 py-1 text-[10px] font-bold uppercase tracking-wider transition-colors border ${
                  shelfFilter === status
                    ? "bg-[#b392ac] text-white border-[#b392ac]"
                    : "bg-white text-gray-400 border-[#eee] hover:border-[#b392ac]"
                }`}
              >
                {status.replace(/_/g, " ")}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[9px] font-bold text-gray-400 uppercase">Sort by</span>
            <select
              value={shelfSort}
              onChange={(e) => setShelfSort(e.target.value)}
              className="text-[10px] bg-transparent border-b border-[#eee] outline-none font-bold text-[#b392ac]"
            >
              <option value="recent">Recently Added</option>
              <option value="rating_high">Rating (High to Low)</option>
              <option value="rating_low">Rating (Low to High)</option>
              <option value="title">Title (A-Z)</option>
            </select>
          </div>
        </div>

        {/* Statistics Card */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white border border-[#eee] p-4 text-center">
            <div className="text-2xl font-bold text-[#b392ac]">{readingStats.total}</div>
            <div className="text-[10px] text-gray-400 uppercase tracking-wider">Total Books</div>
          </div>
          <div className="bg-white border border-[#eee] p-4 text-center">
            <div className="text-2xl font-bold text-[#f4acb7]">{readingStats.want_to_read}</div>
            <div className="text-[10px] text-gray-400 uppercase tracking-wider">Want to Read</div>
          </div>
          <div className="bg-white border border-[#eee] p-4 text-center">
            <div className="text-2xl font-bold text-[#9d7799]">{readingStats.reading}</div>
            <div className="text-[10px] text-gray-400 uppercase tracking-wider">Reading</div>
          </div>
          <div className="bg-white border border-[#eee] p-4 text-center">
            <div className="text-2xl font-bold text-[#735d78]">{readingStats.finished}</div>
            <div className="text-[10px] text-gray-400 uppercase tracking-wider">Finished</div>
          </div>
        </div>

        {/* Mood Preference */}
        <div className="flex items-center gap-4 text-xs font-bold text-[#b392ac] bg-[#e5d9f2]/30 p-4 border border-[#b392ac]/20">
          <BarChart3 className="w-4 h-4" />
          Your collection shows a preference for:{" "}
          {myCollection
            .map((b) => b.mood)
            .filter((v, i, a) => a.indexOf(v) === i)
            .join(", ") || "\u2014"}
        </div>
      </div>

      {/* Book Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
        {filteredBooks.length > 0 ? (
          filteredBooks.map((book, idx) => (
            <BookCard
              key={book.isbn || idx}
              book={book}
              showShelfControls={true}
              isInCollection={true}
              onOpenBook={onOpenBook}
              onRemove={onRemoveBook}
              onRatingChange={onRatingChange}
              onStatusChange={onStatusChange}
            />
          ))
        ) : (
          <div className="col-span-full py-20 text-center text-gray-400 text-xs italic">
            {myCollection.length === 0
              ? "Your bookshelf is empty. Go to Gallery to discover and collect books!"
              : "No books match the current filter."}
          </div>
        )}
      </div>
    </>
  );
};

export default BookshelfPage;
