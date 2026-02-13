import React from "react";
import { X, Search, Loader2 } from "lucide-react";
import { PLACEHOLDER_IMG } from "../constants";

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
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/20 animate-in fade-in">
      <div className="bg-white p-6 shadow-soft border border-[#d3dec7] rounded-2xl w-full max-w-md relative">
        <button onClick={onClose} className="absolute top-2 right-2">
          <X className="w-4 h-4" />
        </button>
        <h3 className="font-bold uppercase tracking-widest mb-4 text-[#5E81AC]">
          Import from Google Books
        </h3>

        <form onSubmit={onSearch} className="flex gap-2 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-2.5 w-4 h-4 text-[#81A1C1]" />
            <input
              autoFocus
              className="w-full border border-[#88C0D0] p-2 pl-8 text-sm outline-none focus:border-[#5E81AC]"
              placeholder="Search title, author, or ISBN..."
              value={googleQuery}
              onChange={(e) => onQueryChange(e.target.value)}
            />
          </div>
          <button
            type="submit"
            disabled={isSearching}
            className="px-4 py-2 text-sm font-bold transition-all bg-[#5E81AC] text-white hover:bg-[#4C566A] disabled:opacity-50"
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
                className="flex gap-3 border border-[#88C0D0] p-2 hover:bg-[#D8DEE9]/30 transition-colors"
              >
                <img src={thumb} className="w-12 h-16 object-cover bg-gray-100" alt="" />
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-bold text-[#4C566A] truncate" title={info.title}>
                    {info.title}
                  </h4>
                  <p className="text-[10px] text-[#5E81AC] truncate">
                    {info.authors?.join(", ")}
                  </p>
                  <p className="text-[10px] text-[#81A1C1] mt-1 line-clamp-2">
                    {info.description}
                  </p>
                </div>
                <button
                  onClick={() => onImport(item)}
                  disabled={!!addingBookId}
                  className="self-center px-3 py-1 bg-[#D08770] text-white text-[10px] font-bold uppercase hover:bg-[#BF616A] disabled:opacity-50"
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
