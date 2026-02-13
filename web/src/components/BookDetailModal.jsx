import React, { useState, useEffect } from "react";
import { X, Sparkles, Info, MessageSquare, MessageCircle, Send, Star, Bookmark } from "lucide-react";
import { getSimilarBooks } from "../api";
import { PLACEHOLDER_IMG } from "../constants";

const BookDetailModal = ({
  book,
  onClose,
  messages,
  onSend,
  input,
  onInputChange,
  myCollection,
  onToggleCollect,
  onRatingChange,
  onStatusChange,
  onUpdateComment,
  onOpenBook,
}) => {
  const [similarBooks, setSimilarBooks] = useState([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  useEffect(() => {
    if (!book?.isbn) return;
    setLoadingSimilar(true);
    getSimilarBooks(book.isbn, 6)
      .then((recs) => {
        const mapped = recs.map((r) => ({
          id: r.isbn,
          title: r.title,
          author: r.authors,
          desc: r.description,
          img: r.thumbnail,
          isbn: r.isbn,
          rating: r.average_rating || 0,
          tags: r.tags || [],
          review_highlights: r.review_highlights || [],
          emotions: r.emotions || {},
          aiHighlight: r.review_highlights?.[0] || "\u2014",
          suggestedQuestions: ["Any similar recommendations?", "What's the core highlight?"],
        }));
        setSimilarBooks(mapped);
      })
      .catch(() => setSimilarBooks([]))
      .finally(() => setLoadingSimilar(false));
  }, [book?.isbn]);

  if (!book) return null;

  const isInCollection = myCollection.some((b) => b.isbn === book.isbn);
  const userBook = myCollection.find((b) => b.isbn === book.isbn);
  const displayRating =
    userBook?.rating && userBook.rating > 0 ? userBook.rating : book.rating || 0;
  const isUserRating = userBook?.rating && userBook.rating > 0;
  const explanations = Array.isArray(book.explanations) ? book.explanations : [];
  const explanationMaxAbs = Math.max(
    1e-9,
    ...explanations.map((e) => Math.abs(Number(e?.contribution ?? 0)))
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/20 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="relative bg-white max-w-5xl w-full rounded-2xl shadow-soft border border-[#d3dec7] my-8"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 text-text-secondary hover:text-text-primary transition-colors z-10 p-1 rounded-full hover:bg-[#F0FDF4]"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="grid md:grid-cols-12 gap-8 md:gap-10 p-8 md:p-10">
          {/* Left Column */}
          <div className="md:col-span-5 flex flex-col items-center md:border-r border-[#d3dec7] pr-0 md:pr-8">
            {/* Cover */}
            <div className="bg-white p-1.5 rounded-xl shadow-soft mb-4 w-52 md:w-56">
              <img
                src={book.img || PLACEHOLDER_IMG}
                alt="cover"
                className="w-full aspect-[3/4] object-cover rounded-lg"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = PLACEHOLDER_IMG;
                }}
              />
            </div>
            <p className="text-xs text-text-secondary mb-1 tracking-tight text-center w-full">
              {book.author}
            </p>
            <h2 className="text-xl font-serif font-bold text-text-primary mb-1 text-center w-full">
              {book.title}
            </h2>
            <p className="text-[10px] text-text-secondary/60 mb-4 text-center w-full">
              ISBN: {book.isbn}
            </p>

            {/* AI Highlight */}
            <div className="bg-tag border border-line p-4 w-full rounded-xl relative mb-4">
              <Sparkles className="w-3 h-3 text-accent absolute -top-1.5 -left-1.5 fill-current" />
              <div className="flex items-center justify-between mb-2">
                <div className="flex flex-col">
                  <span className="text-[12px] font-semibold text-text-primary">
                    {displayRating > 0 ? displayRating.toFixed(1) : "0.0"}
                    <span className="text-text-secondary font-normal ml-1">
                      {isUserRating ? "(Your Rating)" : "(Average)"}
                    </span>
                  </span>
                  <div className="flex gap-0.5 mt-0.5">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <Star
                        key={i}
                        className={`w-3 h-3 ${
                          i <= displayRating ? "text-accent fill-current" : "text-line"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>
              <p className="text-[12px] text-text-secondary italic leading-relaxed">
                {book.aiHighlight}
              </p>
            </div>

            {/* SHAP Explanations */}
            {explanations.length > 0 && (
              <div className="bg-tag border border-line p-4 w-full rounded-xl relative mb-4">
                <Info className="w-3 h-3 text-info absolute -top-1.5 -left-1.5" />
                <p className="text-[10px] font-semibold text-text-primary uppercase tracking-wider mb-3">
                  Why This Recommendation
                </p>
                <div className="space-y-2">
                  {explanations.map((exp, idx) => {
                    const contribution = Number(exp?.contribution ?? 0);
                    const widthPct = Math.min(
                      100,
                      (Math.abs(contribution) / explanationMaxAbs) * 100
                    );
                    const isPositive = exp?.direction === "positive";
                    return (
                      <div
                        key={idx}
                        className="flex items-center gap-2"
                        title={`${exp.feature}: ${contribution.toFixed(4)}`}
                      >
                      <span
                        className={`text-[10px] font-bold w-4 text-center ${
                          isPositive ? "text-info" : "text-danger"
                        }`}
                      >
                        {isPositive ? "+" : "\u2212"}
                      </span>
                      <div className="flex-1 bg-line/40 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isPositive ? "bg-info" : "bg-danger/50"
                          }`}
                          style={{ width: `${widthPct}%` }}
                        />
                      </div>
                      <span className="text-[11px] text-text-secondary font-medium min-w-[120px]">
                        {exp.feature}
                      </span>
                    </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Review Highlights */}
            {book.review_highlights && book.review_highlights.length > 0 && (
              <div className="w-full space-y-2 text-left">
                {book.review_highlights.slice(0, 3).map((highlight, idx) => {
                  const isCompleteSentence = /^[A-Z]/.test(highlight.trim());
                  const prefix = isCompleteSentence ? "" : "...";
                  return (
                    <p key={idx} className="text-[11px] text-text-secondary leading-relaxed italic pl-3 border-l-2 border-line">
                      {prefix}{highlight}
                    </p>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Column */}
          <div className="md:col-span-7 flex flex-col space-y-6">
            {/* Description */}
            <div className="space-y-2">
              <h4 className="flex items-center gap-2 text-[10px] font-semibold uppercase text-text-secondary tracking-wider">
                <Info className="w-3.5 h-3.5" /> Description
              </h4>
              <div className="p-4 bg-tag border border-line text-[13px] leading-relaxed text-text-primary rounded-xl border-l-[3px] border-l-info">
                <div style={{ maxHeight: "180px", overflowY: "auto", whiteSpace: "pre-line" }}>
                  {book.desc}
                </div>
              </div>
            </div>

            {/* Similar Reads */}
            <div className="space-y-2">
              <h4 className="flex items-center gap-2 text-[10px] font-semibold uppercase text-text-secondary tracking-wider">
                Similar Reads
              </h4>
              <div className="flex gap-3 overflow-x-auto pb-2 -mx-1">
                {loadingSimilar ? (
                  <div className="text-[10px] text-text-secondary/60 py-4">Loading similar books...</div>
                ) : similarBooks.length > 0 ? (
                  similarBooks.map((sb) => (
                    <button
                      key={sb.isbn}
                      onClick={() => onOpenBook && onOpenBook(sb)}
                      className="flex-shrink-0 w-16 text-left group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-lg"
                    >
                      <div className="bg-white p-0.5 rounded-lg shadow-soft group-hover:shadow-md transition-shadow">
                        <img
                          src={sb.img || PLACEHOLDER_IMG}
                          alt={sb.title}
                          className="w-full aspect-[3/4] object-cover rounded-md"
                          onError={(e) => { e.target.onerror = null; e.target.src = PLACEHOLDER_IMG; }}
                        />
                      </div>
                      <p className="text-[10px] text-text-secondary mt-1.5 truncate group-hover:text-text-primary transition-colors" title={sb.title}>
                        {sb.title}
                      </p>
                    </button>
                  ))
                ) : (
                  <div className="text-[10px] text-text-secondary/60 py-4">No similar books found</div>
                )}
              </div>
            </div>

            {/* Chat */}
            <div className="flex-grow flex flex-col bg-[#F7F9F2] border border-[#d3dec7] rounded-xl overflow-hidden h-[300px]">
              <div className="p-3 border-b border-[#d3dec7] bg-white flex justify-between items-center">
                <span className="text-[10px] font-semibold text-text-primary flex items-center gap-2 uppercase tracking-wider">
                  <MessageSquare className="w-3 h-3 text-[#5E81AC]" /> Discussion
                </span>
              </div>
              <div className="flex-grow overflow-y-auto p-4 space-y-3">
                <div className="flex justify-start">
                  <div className="max-w-[85%] p-3 bg-[#F0FDF4] border border-[#d3dec7] text-[11px] text-text-primary rounded-2xl rounded-tl-md">
                    Hello! Based on your collection preferences, I found this book&apos;s{" "}
                    {book.mood} atmosphere pairs beautifully with your taste. Would you like to
                    explore its themes?
                  </div>
                </div>
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[80%] p-3 text-[11px] ${
                        m.role === "user"
                          ? "bg-[#5E81AC] text-white rounded-2xl rounded-tr-md"
                          : "bg-[#F0FDF4] text-text-primary border border-[#d3dec7] rounded-2xl rounded-tl-md"
                      }`}
                    >
                      {m.content}
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-3 bg-white border-t border-[#d3dec7] space-y-3">
                <div className="flex flex-wrap gap-2">
                  {(book.suggestedQuestions || []).map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSend(q)}
                      className="text-[10px] px-3 py-1.5 bg-tag border border-line text-text-secondary hover:text-text-primary hover:border-info/40 transition-colors rounded-full"
                    >
                      {q}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    value={input}
                    onChange={(e) => onInputChange(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && onSend(input)}
                    className="flex-grow border border-line p-2.5 text-[12px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background bg-white rounded-full px-4 text-text-primary placeholder:text-text-secondary"
                    placeholder="Ask a question..."
                  />
                  <button
                    onClick={() => onSend(input)}
                    className="bg-[#5E81AC] text-white p-2.5 hover:bg-[#4C566A] transition-colors rounded-full"
                  >
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-3">
              {/* Rating & Status (if in collection) */}
              {isInCollection && (
                <div className="p-4 bg-[#F0FDF4] border border-[#d3dec7] rounded-xl space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-accent uppercase tracking-wider">
                      My Rating
                    </span>
                    <div className="flex gap-0.5">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={star}
                          onClick={() => onRatingChange(book.isbn, star)}
                          className="transform hover:scale-110 transition-transform focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-md"
                        >
                          <Star
                            className={`w-4 h-4 transition-colors ${
                              star <= (userBook?.rating || 0)
                                ? "text-accent fill-current"
                                : "text-line hover:text-accent"
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold text-[#5E81AC] uppercase tracking-wider">
                      Status
                    </span>
                    <select
                      value={userBook?.status || "want_to_read"}
                      onChange={(e) => onStatusChange(book.isbn, e.target.value)}
                      className="bg-white border border-line text-[11px] text-text-primary p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-lg cursor-pointer w-28"
                    >
                      <option value="want_to_read">Want to Read</option>
                      <option value="reading">Reading</option>
                      <option value="finished">Finished</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Collect Button */}
              <button
                onClick={() => onToggleCollect(book)}
                className={`w-full py-3 text-sm font-semibold transition-all flex items-center justify-center gap-2 rounded-xl ${
                  isInCollection
                    ? "bg-accent/10 text-accent border border-accent/30 hover:bg-accent/20"
                    : "bg-[#5E81AC] text-white hover:bg-[#4C566A]"
                }`}
              >
                <Bookmark className={`w-4 h-4 ${isInCollection ? "fill-current" : ""}`} />
                {isInCollection ? "In Collection" : "Add to Collection"}
              </button>

              {/* Notes */}
              {isInCollection && (
                <div className="pt-3 border-t border-line">
                  <label className="text-[10px] font-semibold text-info uppercase tracking-wider mb-2 block flex items-center gap-2">
                    <MessageCircle className="w-3 h-3" /> My Private Notes
                  </label>
                  <textarea
                    value={userBook?.comment || ""}
                    onChange={(e) => onUpdateComment(book.isbn, e.target.value, false)}
                    onBlur={(e) => onUpdateComment(book.isbn, e.target.value, true)}
                    className="w-full text-[12px] p-3 border border-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background h-24 resize-none bg-white text-text-primary placeholder:text-text-secondary rounded-xl"
                    placeholder="Write your thoughts, review, or memorable quotes here..."
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookDetailModal;
