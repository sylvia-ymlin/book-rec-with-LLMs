import React from "react";
import { X, Search, Loader2 } from "lucide-react";

const PLACEHOLDER_IMG = "http://127.0.0.1:6006/assets/cover-not-found.jpg";

const AddBookModal = ({
  onClose,
  googleQuery,
  onQueryChange,
  googleResults,
  isSearching,
  addingBookId,
  onSearch,
  onImport,
}) => {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/10 backdrop-blur-sm animate-in fade-in">
      <div className="bg-white p-6 shadow-xl border border-[#333] w-full max-w-md relative">
        <button onClick={onClose} className="absolute top-2 right-2">
          <X className="w-4 h-4" />
        </button>
        <h3 className="font-bold uppercase tracking-widest mb-4 text-[#b392ac]">
          Import from Google Books
        </h3>

        <form onSubmit={onSearch} className="flex gap-2 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-400" />
            <input
              autoFocus
              className="w-full border p-2 pl-8 text-sm outline-none focus:border-[#b392ac]"
              placeholder="Search title, author, or ISBN..."
              value={googleQuery}
              onChange={(e) => onQueryChange(e.target.value)}
            />
          </div>
          <button
            type="submit"
            disabled={isSearching}
            className="px-4 py-2 text-sm font-bold transition-all bg-[#b392ac] text-white hover:bg-[#9d7799] disabled:opacity-50"
          >
            {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
          </button>
        </form>

        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          {googleResults.length === 0 && !isSearching && googleQuery && (
            <div className="text-center text-gray-400 text-xs py-4">No results found.</div>
          )}

          {googleResults.map((item) => {
            const info = item.volumeInfo;
            const thumb = info.imageLinks?.thumbnail || PLACEHOLDER_IMG;
            return (
              <div
                key={item.id}
                className="flex gap-3 border border-[#eee] p-2 hover:bg-gray-50 transition-colors"
              >
                <img src={thumb} className="w-12 h-16 object-cover bg-gray-100" alt="" />
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-bold text-[#333] truncate" title={info.title}>
                    {info.title}
                  </h4>
                  <p className="text-[10px] text-gray-500 truncate">
                    {info.authors?.join(", ")}
                  </p>
                  <p className="text-[10px] text-gray-400 mt-1 line-clamp-2">
                    {info.description}
                  </p>
                </div>
                <button
                  onClick={() => onImport(item)}
                  disabled={!!addingBookId}
                  className="self-center px-3 py-1 bg-[#b392ac] text-white text-[10px] font-bold uppercase hover:bg-[#9d7799] disabled:opacity-50"
                >
                  {addingBookId === item.id ? "..." : "Import"}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AddBookModal;
