import React from "react";
import { Heart, Star, Trash2 } from "lucide-react";
import { PLACEHOLDER_IMG } from "../constants";

const BookCard = ({
  book,
  showShelfControls = false,
  isInCollection = false,
  onOpenBook,
  onRemove,
  onRatingChange,
  onStatusChange,
}) => {
  const moodColor = (book?.mood || "").toLowerCase();
  const moodClass =
    moodColor === "happy"
      ? "bg-[#D08770]/12 border-[#D08770]/30 text-[#D08770]"
      : moodColor === "sad"
        ? "bg-[#81A1C1]/12 border-[#81A1C1]/30 text-[#5E81AC]"
        : moodColor === "angry"
          ? "bg-[#BF616A]/12 border-[#BF616A]/30 text-[#BF616A]"
          : moodColor === "surprising" || moodColor === "suspenseful"
            ? "bg-[#5E81AC]/10 border-[#5E81AC]/25 text-[#5E81AC]"
            : "bg-[#d3dec7]/30 border-[#d3dec7] text-text-secondary";

  return (
    <div className="group cursor-pointer transform hover:-translate-y-1 transition-all duration-200">
      <div className="bg-white border border-[#d3dec7] p-1.5 relative shadow-soft overflow-hidden rounded-2xl">
        <img
          src={book.img || PLACEHOLDER_IMG}
          alt={book.title}
          className="w-full aspect-[3/4] object-cover rounded-xl opacity-95 group-hover:opacity-100 transition-opacity"
          onClick={() => onOpenBook(book)}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = PLACEHOLDER_IMG;
          }}
        />
        {/* Hover highlight overlay (Discovery mode only) */}
        {!showShelfControls && (
          <div
            className="absolute inset-1.5 rounded-xl bg-white flex items-center justify-center p-4 opacity-0 group-hover:opacity-95 transition-opacity text-center"
            onClick={() => onOpenBook(book)}
          >
            <p className="text-[10px] font-medium text-text-secondary leading-relaxed italic">
              {book.aiHighlight}
            </p>
          </div>
        )}
        {/* Collection badge */}
        {isInCollection && (
          <div className="absolute top-2.5 right-2.5 bg-[#BF616A] p-1 rounded-full shadow-sm">
            <Heart className="w-2.5 h-2.5 text-white fill-current" />
          </div>
        )}
        {/* Rank Badge - Discovery mode only */}
        {!showShelfControls && book.rank && (
          <div className="absolute top-2.5 left-2.5 bg-text-primary text-white text-[10px] font-bold px-2 py-0.5 rounded-full">
            #{book.rank}
          </div>
        )}
        {/* Remove button - Bookshelf mode only */}
        {showShelfControls && onRemove && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemove(book.isbn);
            }}
            className="absolute top-2.5 left-2.5 bg-[#BF616A] p-1 rounded-full shadow-sm opacity-0 group-hover:opacity-100 transition-opacity hover:bg-[#D08770]"
            title="Remove from collection"
          >
            <Trash2 className="w-2.5 h-2.5 text-white" />
          </button>
        )}
      </div>
      <h3
        className="mt-3 text-[13px] font-serif font-semibold text-text-primary truncate"
        onClick={() => onOpenBook(book)}
      >
        {book.title}
      </h3>
      <div className="flex justify-between items-center mt-1">
        <div className="flex flex-col">
          <span className="text-[10px] text-text-secondary tracking-tight truncate w-24">
            {book.author}
          </span>
          {!showShelfControls && book.rating > 0 && (
            <div className="flex items-center gap-0.5 mt-0.5">
              <Star className="w-2.5 h-2.5 text-accent fill-current" />
              <span className="text-[9px] font-semibold text-accent">
                {book.rating.toFixed(1)}
              </span>
            </div>
          )}
        </div>
        {book.emotions && Object.keys(book.emotions).length > 0 ? (
          <span className={`text-[9px] border px-2 py-0.5 rounded-full capitalize ${moodClass}`}>
            {Object.entries(book.emotions).reduce((a, b) => (a[1] > b[1] ? a : b))[0]}
          </span>
        ) : (
          <span className="text-[9px] bg-[#F0FDF4] border border-[#d3dec7] px-2 py-0.5 rounded-full text-text-secondary">&mdash;</span>
        )}
      </div>

      {/* Rating and Status for Bookshelf View */}
      {showShelfControls && (
        <div className="mt-2 space-y-2">
          {/* Star Rating */}
          <div className="flex gap-0.5">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={(e) => {
                  e.stopPropagation();
                  onRatingChange && onRatingChange(book.isbn, star);
                }}
                className="focus:outline-none"
              >
                <Star
                  className={`w-3.5 h-3.5 transition-colors ${
                    star <= (book.rating || 0)
                      ? "text-accent fill-current"
                      : "text-[#d3dec7] hover:text-accent"
                  }`}
                />
              </button>
            ))}
          </div>
          {/* Status Dropdown */}
          <select
            value={book.status || "want_to_read"}
            onChange={(e) => {
              e.stopPropagation();
              onStatusChange && onStatusChange(book.isbn, e.target.value);
            }}
            onClick={(e) => e.stopPropagation()}
            className="w-full text-[9px] p-1.5 border border-[#d3dec7] bg-white text-text-primary outline-none focus:border-[#5E81AC]/60 rounded-lg"
          >
            <option value="want_to_read">Want to Read</option>
            <option value="reading">Reading</option>
            <option value="finished">Finished</option>
          </select>
        </div>
      )}
    </div>
  );
};

export default BookCard;
