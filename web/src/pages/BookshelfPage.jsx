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

  const statusAccent = {
    all: "bg-[#5E81AC] text-white border-[#5E81AC] shadow-sm",
    want_to_read: "bg-[#D08770] text-white border-[#D08770] shadow-sm",
    reading: "bg-[#81A1C1] text-white border-[#81A1C1] shadow-sm",
    finished: "bg-[#4C566A] text-white border-[#4C566A] shadow-sm",
  };

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
        <div className="flex justify-between items-center bg-white p-4 border border-[#d0dcc2] shadow-soft mb-4 rounded-[32px]">
          <div className="flex gap-2">
            {["all", "want_to_read", "reading", "finished"].map((status) => (
              <button
                key={status}
                onClick={() => setShelfFilter(status)}
                className={`px-4 py-2 text-[10px] font-semibold uppercase tracking-wider transition-all border rounded-full ${
                  shelfFilter === status
                    ? statusAccent[status]
                    : "bg-white text-[#81A1C1] border-[#88C0D0] hover:border-[#5E81AC] hover:text-[#5E81AC]"
                }`}
              >
                {status.replace(/_/g, " ")}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold text-text-secondary uppercase">Sort by</span>
            <select
              value={shelfSort}
              onChange={(e) => setShelfSort(e.target.value)}
              className="text-[10px] bg-transparent border-b border-[#d3dfc8] outline-none font-semibold text-text-primary py-1"
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
          <div className="bg-white border border-[#d3dfc8] p-4 text-center rounded-3xl shadow-soft">
            <div className="text-2xl font-bold text-[#5E81AC]">{readingStats.total}</div>
            <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider">Total Books</div>
          </div>
          <div className="bg-white border border-[#E5C49D] p-4 text-center rounded-3xl shadow-soft">
            <div className="text-2xl font-bold text-[#D08770]">{readingStats.want_to_read}</div>
            <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider">Want to Read</div>
          </div>
          <div className="bg-white border border-[#c8d9d4] p-4 text-center rounded-3xl shadow-soft">
            <div className="text-2xl font-bold text-[#81A1C1]">{readingStats.reading}</div>
            <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider">Reading</div>
          </div>
          <div className="bg-white border border-[#d3dfc8] p-4 text-center rounded-3xl shadow-soft">
            <div className="text-2xl font-bold text-[#4C566A]">{readingStats.finished}</div>
            <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider">Finished</div>
          </div>
        </div>

        {/* Mood Preference */}
        <div className="flex items-center gap-4 text-xs font-semibold text-text-primary bg-[#F0FDF4] p-4 border border-[#d4e2cb] rounded-[32px] shadow-soft">
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
