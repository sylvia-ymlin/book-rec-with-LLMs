import React from "react";
import { Heart, Star, Trash2 } from "lucide-react";

const PLACEHOLDER_IMG = "http://127.0.0.1:6006/assets/cover-not-found.jpg";

const BookCard = ({
  book,
  showShelfControls = false,
  isInCollection = false,
  onOpenBook,
  onRemove,
  onRatingChange,
  onStatusChange,
}) => {
  return (
    <div className="group cursor-pointer transform hover:-translate-y-1 transition-all">
      <div className="bg-white border border-[#eee] p-1 relative shadow-sm group-hover:shadow-md overflow-hidden">
        <img
          src={book.img || PLACEHOLDER_IMG}
          alt={book.title}
          className="w-full aspect-[3/4] object-cover opacity-90 group-hover:opacity-100 transition-opacity"
          onClick={() => onOpenBook(book)}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = PLACEHOLDER_IMG;
          }}
        />
        {/* Hover highlight overlay (Discovery mode only) */}
        {!showShelfControls && (
          <div
            className="absolute inset-0 bg-white/80 flex items-center justify-center p-4 opacity-0 group-hover:opacity-100 transition-opacity text-center px-4"
            onClick={() => onOpenBook(book)}
          >
            <p className="text-[10px] font-bold text-[#b392ac] leading-relaxed italic">
              {book.aiHighlight}
            </p>
          </div>
        )}
        {/* Collection badge */}
        {isInCollection && (
          <div className="absolute top-1 right-1 bg-[#f4acb7] p-1 shadow-sm">
            <Heart className="w-3 h-3 text-white fill-current" />
          </div>
        )}
        {/* Rank Badge - Discovery mode only */}
        {!showShelfControls && book.rank && (
          <div className="absolute top-1 left-1 bg-black/70 text-white text-[10px] font-bold px-1.5 py-0.5 shadow-sm z-10 backdrop-blur-sm">
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
            className="absolute top-1 left-1 bg-red-400 p-1 shadow-sm opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"
            title="Remove from collection"
          >
            <Trash2 className="w-3 h-3 text-white" />
          </button>
        )}
      </div>
      <h3
        className="mt-3 text-[12px] font-bold text-[#555] truncate"
        onClick={() => onOpenBook(book)}
      >
        {book.title}
      </h3>
      <div className="flex justify-between items-center mt-1">
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 tracking-tighter truncate w-24">
            {book.author}
          </span>
          {!showShelfControls && book.rating > 0 && (
            <div className="flex items-center gap-0.5 mt-0.5">
              <Star className="w-2 h-2 text-[#f4acb7] fill-current" />
              <span className="text-[8px] font-bold text-[#f4acb7]">
                {book.rating.toFixed(1)}
              </span>
            </div>
          )}
        </div>
        {book.emotions && Object.keys(book.emotions).length > 0 ? (
          <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999] capitalize">
            {Object.entries(book.emotions).reduce((a, b) => (a[1] > b[1] ? a : b))[0]}
          </span>
        ) : (
          <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999]">&mdash;</span>
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
                      ? "text-[#f4acb7] fill-current"
                      : "text-gray-200 hover:text-[#f4acb7]"
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
            className="w-full text-[9px] p-1 border border-[#eee] bg-white text-gray-500 outline-none focus:border-[#b392ac]"
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
