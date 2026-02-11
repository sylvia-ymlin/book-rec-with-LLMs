import React from "react";
import { X, Sparkles, Info, MessageSquare, MessageCircle, Send, Star, Bookmark } from "lucide-react";

const PLACEHOLDER_IMG = "/content/cover-not-found.jpg";

const StudyCard = ({ children, className }) => (
  <div className={`bg-white border-2 border-[#333] shadow-md ${className || ""}`}>
    {children}
  </div>
);

const StudyButton = ({ children, active, color, className, onClick }) => {
  const colors = {
    purple: "bg-[#b392ac] text-white hover:bg-[#9d7799]",
    peach: "bg-[#f4acb7] text-white hover:bg-[#e89ba3]",
  };
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-bold transition-all ${colors[color] || colors.purple} ${className || ""}`}
    >
      {children}
    </button>
  );
};

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
}) => {
  if (!book) return null;

  const isInCollection = myCollection.some((b) => b.isbn === book.isbn);
  const userBook = myCollection.find((b) => b.isbn === book.isbn);
  const displayRating =
    userBook?.rating && userBook.rating > 0 ? userBook.rating : book.rating || 0;
  const isUserRating = userBook?.rating && userBook.rating > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/5 backdrop-blur-sm animate-in fade-in duration-300 overflow-y-auto">
      <StudyCard className="relative bg-white max-w-5xl w-full shadow-2xl border-[#333] my-8">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-300 hover:text-gray-600 transition-colors z-10"
        >
          <X className="w-6 h-6" />
        </button>

        <div className="grid md:grid-cols-12 gap-8 md:gap-10 px-6 md:px-10 py-6">
          {/* Left Column */}
          <div className="md:col-span-5 flex flex-col items-center border-r border-[#f5f5f5] pr-0 md:pr-6">
            <div className="border border-[#eee] p-1 bg-white shadow-sm mb-2 w-52 md:w-56">
              <img
                src={book.img || PLACEHOLDER_IMG}
                alt="cover"
                className="w-full aspect-[3/4] object-cover"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = PLACEHOLDER_IMG;
                }}
              />
            </div>
            <p className="text-xs text-[#999] mb-2 tracking-tighter text-center w-full">
              {book.author}
            </p>
            <h2 className="text-xl font-bold text-[#333] mb-1 text-center md:text-left w-full">
              {book.title}
            </h2>
            <p className="text-xs text-[#999] mb-2 tracking-tighter text-center md:text-left w-full">
              ISBN: {book.isbn}
            </p>

            {/* AI Highlight Box */}
            <div className="bg-[#fff9f9] border border-[#f4acb7] p-4 w-full relative mb-4">
              <Sparkles className="w-3 h-3 text-[#f4acb7] absolute -top-1.5 -left-1.5 fill-current" />
              <div className="flex items-center justify-between mb-2">
                <div className="flex flex-col">
                  <span className="text-[11px] font-bold text-[#f4acb7]">
                    {displayRating > 0 ? displayRating.toFixed(1) : "0.0"}
                    {isUserRating ? " (Your Rating)" : " (Average)"}
                  </span>
                  <div className="flex gap-0.5 text-[#f4acb7]">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <Star key={i} className={`w-3 h-3 ${i <= displayRating ? "fill-current" : ""}`} />
                    ))}
                  </div>
                </div>
              </div>
              <p className="text-[11px] font-bold text-[#f4acb7] italic leading-relaxed">
                {book.aiHighlight}
              </p>
            </div>

            {/* Why This Recommendation — SHAP Explanations (V2.7) */}
            {book.explanations && book.explanations.length > 0 && (
              <div className="bg-[#f8f5ff] border border-[#b392ac]/40 p-4 w-full relative mb-4">
                <Info className="w-3 h-3 text-[#b392ac] absolute -top-1.5 -left-1.5" />
                <p className="text-[11px] font-bold text-[#b392ac] uppercase tracking-wider mb-3">
                  Why This Recommendation
                </p>
                <div className="space-y-2">
                  {book.explanations.map((exp, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span
                        className={`text-[9px] font-bold w-4 text-center ${exp.direction === "positive" ? "text-[#b392ac]" : "text-gray-400"
                          }`}
                      >
                        {exp.direction === "positive" ? "+" : "\u2212"}
                      </span>
                      <div className="flex-1 bg-gray-100 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${exp.direction === "positive"
                              ? "bg-gradient-to-r from-[#b392ac] to-[#9d7799]"
                              : "bg-gray-300"
                            }`}
                          style={{
                            width: `${Math.min(Math.abs(exp.contribution) * 150, 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-[#555] font-medium min-w-[100px]">
                        {exp.feature}
                      </span>
                    </div>
                  ))}
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
                    <p key={idx} className="text-[10px] text-[#666] leading-relaxed italic pl-2">
                      - &ldquo;{prefix}{highlight}&rdquo;
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
              <h4 className="flex items-center gap-2 text-[10px] font-bold uppercase text-gray-400 tracking-wider">
                <Info className="w-3.5 h-3.5" /> Description
              </h4>
              <div className="p-4 bg-white border border-[#eee] text-[12px] leading-relaxed text-[#666] italic border-l-[4px] border-l-[#b392ac]">
                <div style={{ maxHeight: "180px", overflowY: "auto", whiteSpace: "pre-line" }}>
                  {book.desc}
                </div>
              </div>
            </div>

            {/* Chat */}
            <div className="flex-grow flex flex-col border border-[#eee] bg-[#faf9f6] overflow-hidden h-[300px]">
              <div className="p-2 border-b border-[#eee] bg-white flex justify-between items-center">
                <span className="text-[10px] font-bold text-[#b392ac] flex items-center gap-2 uppercase tracking-widest">
                  <MessageSquare className="w-3 h-3" /> Discussion
                </span>
              </div>
              <div className="flex-grow overflow-y-auto p-4 space-y-3">
                <div className="flex justify-start">
                  <div className="max-w-[85%] p-2 bg-white border border-[#eee] text-[11px] text-[#735d78] shadow-sm">
                    Hello! Based on your collection preferences, I found this book&apos;s{" "}
                    {book.mood} atmosphere pairs beautifully with your taste. Would you like to
                    explore its themes?
                  </div>
                </div>
                {messages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[80%] p-2 border text-[11px] shadow-sm ${m.role === "user"
                          ? "bg-[#b392ac] text-white border-[#b392ac]"
                          : "bg-white text-[#666] border-[#eee]"
                        }`}
                    >
                      {m.content}
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-3 bg-white border-t border-[#eee] space-y-3">
                <div className="flex flex-wrap gap-2">
                  {(book.suggestedQuestions || []).map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSend(q)}
                      className="text-[9px] px-2 py-1 bg-[#f8f9fa] border border-[#eee] text-gray-500 hover:border-[#b392ac] hover:text-[#b392ac] transition-colors"
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
                    className="flex-grow border border-[#eee] p-2 text-[11px] outline-none focus:border-[#b392ac] bg-[#faf9f6] font-serif"
                    placeholder="Ask a question..."
                  />
                  <button onClick={() => onSend(input)} className="bg-[#333] text-white p-2">
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-3">
              {/* Rating & Status (if in collection) */}
              {isInCollection && (
                <div className="p-3 bg-[#fff9f9] border border-[#f4acb7] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-[#f4acb7] uppercase tracking-wider">
                      My Rating
                    </span>
                    <div className="flex gap-0.5">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={star}
                          onClick={() => onRatingChange(book.isbn, star)}
                          className="focus:outline-none transform hover:scale-110 transition-transform"
                        >
                          <Star
                            className={`w-4 h-4 transition-colors ${star <= (userBook?.rating || 0)
                                ? "text-[#f4acb7] fill-current"
                                : "text-gray-200 hover:text-[#f4acb7]"
                              }`}
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-[#b392ac] uppercase tracking-wider">
                      Status
                    </span>
                    <select
                      value={userBook?.status || "want_to_read"}
                      onChange={(e) => onStatusChange(book.isbn, e.target.value)}
                      className="bg-white border border-[#eee] text-[10px] text-gray-500 p-1 outline-none focus:border-[#b392ac] w-28 cursor-pointer"
                    >
                      <option value="want_to_read">Want to Read</option>
                      <option value="reading">Reading</option>
                      <option value="finished">Finished</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Collect Button */}
              <StudyButton
                active
                color={isInCollection ? "peach" : "purple"}
                className="w-full py-3 text-sm flex items-center justify-center gap-2 font-bold transition-all"
                onClick={() => onToggleCollect(book)}
              >
                <Bookmark className={`w-4 h-4 ${isInCollection ? "fill-current" : ""}`} />
                {isInCollection ? "In Collection" : "Add to Collection"}
              </StudyButton>

              {/* Notes */}
              {isInCollection && (
                <div className="mt-2 pt-3 border-t border-[#eee]">
                  <label className="text-[10px] font-bold text-[#b392ac] uppercase tracking-wider mb-2 block flex items-center gap-2">
                    <MessageCircle className="w-3 h-3" /> My Private Notes
                  </label>
                  <textarea
                    value={userBook?.comment || ""}
                    onChange={(e) => onUpdateComment(book.isbn, e.target.value, false)}
                    onBlur={(e) => onUpdateComment(book.isbn, e.target.value, true)}
                    className="w-full text-[11px] p-3 border border-[#eee] focus:border-[#b392ac] outline-none h-24 resize-none bg-[#fff9f9] text-[#666] placeholder:text-gray-300 shadow-inner"
                    placeholder="Write your thoughts, review, or memorable quotes here..."
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </StudyCard>
    </div>
  );
};

export default BookDetailModal;
