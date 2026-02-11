import React, { useState } from "react";
import { Bookmark, Heart, Search, Layers, Smile, Sparkles, Star, Trophy, BarChart3, X, MessageCircle, MessageSquare, Info, Send } from "lucide-react";
import { recommend, addFavorite, getPersona, getHighlights } from "./api";

// --- Elegant Book Discovery UI ---

const CATEGORIES = ["All", "Fiction", "History", "Philosophy", "Science", "Art"];
const MOODS = ["All", "Happy", "Suspenseful", "Angry", "Sad", "Surprising"];

const StudyButton = ({ children, active, color, className, onClick }) => {
  const colors = {
    purple: "bg-[#b392ac] text-white hover:bg-[#9d7799]",
    peach: "bg-[#f4acb7] text-white hover:bg-[#e89ba3]",
    tab: "bg-transparent text-[#b392ac] border-b-2 border-[#b392ac]",
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

const StudyCard = ({ children, className }) => (
  <div className={`bg-white border-2 border-[#333] shadow-md ${className || ""}`}>
    {children}
  </div>
);

const App = () => {
  const [selectedBook, setSelectedBook] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [myCollection, setMyCollection] = useState([]); 
  const [showMyShelf, setShowMyShelf] = useState(false);
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  const [searchQuery, setSearchQuery] = useState("");
  const [searchCategory, setSearchCategory] = useState("All");
  const [searchMood, setSearchMood] = useState("All");

  const handleSend = (text) => {
    if (!text) return;
    const newMsgs = [...messages, { role: 'user', content: text }];
    setMessages(newMsgs);
    setInput("");
    setTimeout(() => {
      setMessages([...newMsgs, { role: 'ai', content: `Based on "${selectedBook?.title || ''}" and your reading taste, I recommend paying attention to the thematic elements—they truly resonate with your preferences.` }]);
    }, 600);
  };

  const toggleCollect = async (book) => {
    try {
      await addFavorite(book.isbn);
      if (myCollection.some(b => b.isbn === book.isbn)) {
        setMyCollection(myCollection.filter(b => b.isbn !== book.isbn));
      } else {
        setMyCollection([...myCollection, book]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const openBook = async (book) => {
    setSelectedBook(book);
    setMessages([]);
    try {
      const res = await getHighlights(book.isbn);
      const meta = res?.meta || {};
      setSelectedBook({ ...book, aiHighlight: (res?.highlights || []).join("\n") || '—', suggestedQuestions: [
        `Who is the target audience for this book?`,
        `Does the author have similar works?`,
        `Can you summarize the main content?`
      ], desc: meta?.description || book.desc });
    } catch (e) {
      // keep default
    }
  };

  const startDiscovery = async () => {
    setLoading(true); 
    setError("");
    try {
      const recs = await recommend(searchQuery || 'adventure', searchCategory, searchMood);
      const mapped = (recs || []).map((r, idx) => ({
        id: r.isbn,
        title: r.title,
        author: r.authors,
        category: searchCategory,
        mood: searchMood,
        rank: idx + 1,
        rating: r.average_rating || 0,
        tags: r.tags || [],
        review_highlights: r.review_highlights || [],
        desc: r.description,
        img: r.thumbnail,
        isbn: r.isbn,
        emotions: r.emotions || {},
        aiHighlight: '—',
        suggestedQuestions: [
          `Matches my current mood?`,
          `Any similar recommendations?`,
          `What's the core highlight?`
        ]
      }));
      setBooks(mapped);
    } catch (e) {
      setError(e.message || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const getRecommendedBooks = () => {
    if (myCollection.length === 0) return books.slice(0, 3);
    return books.filter(b => !myCollection.some(cb => cb.isbn === b.isbn)).slice(0, 3);
  };

  const currentViewBooks = showMyShelf ? myCollection : books;

  return (
    <div className="min-h-screen bg-[#faf9f6] text-[#444] font-serif tracking-tight">
      <header className="max-w-5xl mx-auto pt-10 px-4 flex justify-between items-end mb-12">
        <div>
          <div className="border border-[#333] px-4 py-1 bg-white shadow-[2px_2px_0px_0px_#eee] inline-block mb-2">
            <h1 className="text-xl font-bold uppercase tracking-[0.2em] text-[#333]">Paper Shelf</h1>
          </div>
          <p className="text-[10px] text-gray-400 font-medium tracking-widest">Discover books that resonate with your soul</p>
        </div>
        <div className="flex gap-2">
          <StudyButton 
            active={showMyShelf} 
            color={showMyShelf ? "purple" : "tab"}
            onClick={() => setShowMyShelf(!showMyShelf)}
          >
            <Bookmark className="w-4 h-4 inline mr-1" /> {showMyShelf ? "Back to Gallery" : "My Collection"}
          </StudyButton>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 pb-20">
        {!showMyShelf && (
          <>
            {myCollection.length > 0 && (
              <div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
                <h4 className="flex items-center gap-2 text-[10px] font-black uppercase text-[#b392ac] mb-4 tracking-widest">
                  <Sparkles className="w-3.5 h-3.5" /> Soul-Matched Recommendations
                </h4>
                <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide">
                  {getRecommendedBooks().map(book => (
                    <div 
                      key={book.id} 
                      onClick={() => openBook(book)}
                      className="min-w-[280px] flex gap-4 bg-white border border-[#333] p-3 shadow-sm hover:shadow-md cursor-pointer transition-all"
                    >
                      <img src={book.img} className="w-20 h-28 object-cover border border-[#eee]" />
                      <div className="flex flex-col justify-between">
                        <div>
                          <h5 className="text-[12px] font-bold text-[#333]">{book.title}</h5>
                          <p className="text-[10px] text-gray-400 mt-1">Resonates with your "{book.mood}" preference</p>
                        </div>
                        <div className="flex gap-1">
                          {book.tags.slice(0, 2).map(t => <span key={t} className="text-[8px] px-1.5 py-0.5 bg-[#f8f9fa] border border-[#eee] text-[#999]">{t}</span>)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="max-w-4xl mx-auto mb-16 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                <div className="md:col-span-6 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
                  <Search className="w-4 h-4 mr-3 text-gray-300 ml-2" />
                  <input 
                    className="w-full outline-none text-sm placeholder-gray-400 bg-transparent font-serif"
                    placeholder="Search for a topic, mood, or dream..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <div className="md:col-span-3 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
                  <Layers className="w-4 h-4 mr-3 text-gray-300 ml-2" />
                  <select 
                    className="w-full outline-none text-sm bg-transparent text-gray-500 font-serif"
                    value={searchCategory}
                    onChange={(e) => setSearchCategory(e.target.value)}
                  >
                    {CATEGORIES.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                  </select>
                </div>
                <div className="md:col-span-3 flex items-center bg-white border border-[#ddd] p-2 shadow-sm">
                  <Smile className="w-4 h-4 mr-3 text-gray-300 ml-2" />
                  <select 
                    className="w-full outline-none text-sm bg-transparent text-gray-500 font-serif"
                    value={searchMood}
                    onChange={(e) => setSearchMood(e.target.value)}
                  >
                    {MOODS.map(mood => <option key={mood} value={mood}>{mood}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex justify-center">
                <StudyButton active color="purple" className="px-12 py-2" onClick={startDiscovery}>
                  Start Discovery
                </StudyButton>
              </div>
              {loading && <div className="text-center text-xs text-gray-400">Loading...</div>}
              {error && <div className="text-center text-xs text-red-400">{error}</div>}
            </div>
          </>
        )}

        {showMyShelf && (
          <div className="mb-8 flex items-center gap-4 text-xs font-bold text-[#b392ac] bg-[#e5d9f2]/30 p-4 border border-[#b392ac]/20">
            <BarChart3 className="w-4 h-4" /> 
            Your collection shows a preference for: {myCollection.map(b => b.mood).filter((v, i, a) => a.indexOf(v) === i).join(", ")}
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {currentViewBooks.length > 0 ? currentViewBooks.map((book, idx) => (
            <div 
              key={idx}
              onClick={() => openBook(book)}
              className="group cursor-pointer transform hover:-translate-y-1 transition-all"
            >
              <div className="bg-white border border-[#eee] p-1 relative shadow-sm group-hover:shadow-md overflow-hidden">
                <img src={book.img} alt={book.title} className="w-full aspect-[3/4] object-cover opacity-90 group-hover:opacity-100 transition-opacity" />
                <div className="absolute inset-0 bg-white/80 flex items-center justify-center p-4 opacity-0 group-hover:opacity-100 transition-opacity text-center px-4">
                  <p className="text-[10px] font-bold text-[#b392ac] leading-relaxed italic">
                    {book.aiHighlight}
                  </p>
                </div>
                {myCollection.some(b => b.isbn === book.isbn) && (
                  <div className="absolute top-1 right-1 bg-[#f4acb7] p-1 shadow-sm">
                    <Heart className="w-3 h-3 text-white fill-current" />
                  </div>
                )}
              </div>
              <h3 className="mt-3 text-[12px] font-bold text-[#555] truncate">{book.title}</h3>
              <div className="flex justify-between items-center mt-1">
                <span className="text-[9px] text-gray-400 tracking-tighter">{book.author}</span>
                {book.emotions && Object.keys(book.emotions).length > 0 ? (
                  <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999] capitalize">
                    {Object.entries(book.emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0]}
                  </span>
                ) : (
                  <span className="text-[9px] bg-[#f8f9fa] border border-[#eee] px-1 text-[#999]">—</span>
                )}
              </div>
            </div>
          )) : (
            <div className="col-span-full py-20 text-center text-gray-400 text-xs italic">
              No books here yet. Start discovering to build your collection.
            </div>
          )}
        </div>

        {selectedBook && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/5 backdrop-blur-sm animate-in fade-in duration-300 overflow-y-auto">
            <StudyCard className="relative bg-white max-w-5xl w-full shadow-2xl border-[#333] my-8">
              <button 
                onClick={() => setSelectedBook(null)}
                className="absolute top-4 right-4 text-gray-300 hover:text-gray-600 transition-colors z-10"
              >
                <X className="w-6 h-6" />
              </button>

              <div className="grid md:grid-cols-12 gap-8 md:gap-10 px-6 md:px-10 py-6">
                <div className="md:col-span-5 flex flex-col items-center border-r border-[#f5f5f5] pr-0 md:pr-6">
                  <div className="border border-[#eee] p-1 bg-white shadow-sm mb-2 w-52 md:w-56">
                    <img src={selectedBook.img} alt="cover" className="w-full aspect-[3/4] object-cover" />
                  </div>
                  
                  <p className="text-xs text-[#999] mb-2 tracking-tighter text-center w-full">{selectedBook.author}</p>
                
                  <h2 className="text-xl font-bold text-[#333] mb-1 text-center md:text-left w-full">{selectedBook.title}</h2>
                  <p className="text-xs text-[#999] mb-2 tracking-tighter text-center md:text-left w-full">ISBN: {selectedBook.isbn}</p>
                  
                  <div className="bg-[#fff9f9] border border-[#f4acb7] p-4 w-full relative mb-4">
                    <Sparkles className="w-3 h-3 text-[#f4acb7] absolute -top-1.5 -left-1.5 fill-current" />
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-bold text-[#f4acb7]">{selectedBook.rating ? selectedBook.rating.toFixed(1) : '0.0'}</span>
                      <div className="flex gap-0.5 text-[#f4acb7]">
                        {[1,2,3,4,5].map(i => <Star key={i} className={`w-3 h-3 ${i <= selectedBook.rating ? 'fill-current' : ''}`} />)}
                      </div>
                    </div>
                    <p className="text-[11px] font-bold text-[#f4acb7] italic leading-relaxed">
                      {selectedBook.aiHighlight}
                    </p>
                  </div>
                  
                  {selectedBook.review_highlights && selectedBook.review_highlights.length > 0 && (
                    <div className="w-full space-y-2 text-left">
                      {selectedBook.review_highlights.slice(0, 3).map((highlight, idx) => {
                        const isCompleteSentence = /^[A-Z]/.test(highlight.trim());
                        const prefix = isCompleteSentence ? '' : '...';
                        return (
                          <p key={idx} className="text-[10px] text-[#666] leading-relaxed italic pl-2">
                            - "{prefix}{highlight}"
                          </p>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="md:col-span-7 flex flex-col space-y-6">
                  <div className="space-y-2">
                    <h4 className="flex items-center gap-2 text-[10px] font-bold uppercase text-gray-400 tracking-wider">
                      <Info className="w-3.5 h-3.5" /> Summary
                    </h4>
                    <div className="p-4 bg-white border border-[#eee] text-[12px] leading-relaxed text-[#666] italic border-l-[4px] border-l-[#b392ac]">
                      "{selectedBook.desc}"
                    </div>
                  </div>

                  <div className="flex-grow flex flex-col border border-[#eee] bg-[#faf9f6] overflow-hidden h-[300px]">
                    <div className="p-2 border-b border-[#eee] bg-white flex justify-between items-center">
                      <span className="text-[10px] font-bold text-[#b392ac] flex items-center gap-2 uppercase tracking-widest">
                        <MessageSquare className="w-3 h-3" /> Discussion
                      </span>
                    </div>
                    <div className="flex-grow overflow-y-auto p-4 space-y-3">
                      <div className="flex justify-start">
                        <div className="max-w-[85%] p-2 bg-white border border-[#eee] text-[11px] text-[#735d78] shadow-sm">
                          Hello! Based on your collection preferences, I found this book's {selectedBook.mood} atmosphere pairs beautifully with your taste. Would you like to explore its themes?
                        </div>
                      </div>
                      {messages.map((m, i) => (
                        <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] p-2 border text-[11px] shadow-sm ${
                            m.role === 'user' 
                              ? 'bg-[#b392ac] text-white border-[#b392ac]' 
                              : 'bg-white text-[#666] border-[#eee]'
                          }`}>
                            {m.content}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="p-3 bg-white border-t border-[#eee] space-y-3">
                      <div className="flex flex-wrap gap-2">
                        {(selectedBook.suggestedQuestions || []).map((q, idx) => (
                          <button 
                            key={idx}
                            onClick={() => handleSend(q)}
                            className="text-[9px] px-2 py-1 bg-[#f8f9fa] border border-[#eee] text-gray-500 hover:border-[#b392ac] hover:text-[#b392ac] transition-colors"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <input 
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSend(input)}
                          className="flex-grow border border-[#eee] p-2 text-[11px] outline-none focus:border-[#b392ac] bg-[#faf9f6] font-serif" 
                          placeholder="Ask a question..." 
                        />
                        <button onClick={() => handleSend(input)} className="bg-[#333] text-white p-2">
                          <Send className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <StudyButton 
                      active 
                      color={myCollection.some(b => b.isbn === selectedBook.isbn) ? "peach" : "purple"} 
                      className="flex-grow py-3 text-sm flex items-center justify-center gap-2 font-bold"
                      onClick={() => toggleCollect(selectedBook)}
                    >
                      <Bookmark className={`w-4 h-4 ${myCollection.some(b => b.isbn === selectedBook.isbn) ? 'fill-current' : ''}`} />
                      {myCollection.some(b => b.isbn === selectedBook.isbn) ? "In Collection" : "Add to Collection"}
                    </StudyButton>
                  </div>
                </div>
              </div>
            </StudyCard>
          </div>
        )}
      </main>

      <footer className="mt-16 text-center text-[9px] font-medium text-gray-300 uppercase tracking-widest pb-10 border-t border-[#eee] pt-10">
        Paper Shelf // 2026 Your Personal Library
      </footer>
    </div>
  );
};

export default App;
