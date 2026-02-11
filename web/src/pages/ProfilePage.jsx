import React, { useState, useEffect } from "react";
import { UserCircle, BookOpen, Star, Target, TrendingUp, Clock, Award, BarChart3 } from "lucide-react";
import { getPersona } from "../api";

const PLACEHOLDER_IMG = "/content/cover-not-found.jpg";

const ProfilePage = ({ userId, myCollection, readingStats }) => {
  const [persona, setPersona] = useState(null);
  const [loadingPersona, setLoadingPersona] = useState(false);

  useEffect(() => {
    if (!userId) return;
    setLoadingPersona(true);
    getPersona(userId)
      .then((data) => setPersona(data))
      .catch(() => setPersona(null))
      .finally(() => setLoadingPersona(false));
  }, [userId, myCollection.length]);

  // Compute reading insights from collection
  const ratingDistribution = [1, 2, 3, 4, 5].map((star) => ({
    star,
    count: myCollection.filter((b) => Math.round(b.rating || 0) === star).length,
  }));
  const maxRatingCount = Math.max(...ratingDistribution.map((r) => r.count), 1);

  const avgRating =
    myCollection.length > 0
      ? (
        myCollection.reduce((sum, b) => sum + (b.rating || 0), 0) /
        myCollection.filter((b) => b.rating > 0).length || 0
      ).toFixed(1)
      : "0.0";

  const completionRate =
    readingStats.total > 0
      ? Math.round((readingStats.finished / readingStats.total) * 100)
      : 0;

  const recentlyFinished = myCollection
    .filter((b) => b.status === "finished")
    .slice(-5)
    .reverse();

  return (
    <div className="space-y-8">
      {/* Profile Header Card */}
      <div className="bg-white border border-[#eee] p-8 shadow-sm">
        <div className="flex items-start gap-6">
          <div className="w-20 h-20 bg-gradient-to-br from-[#b392ac] to-[#735d78] rounded-full flex items-center justify-center shadow-md">
            <UserCircle className="w-10 h-10 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-[#333] mb-1">Reader Profile</h2>
            <p className="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4">
              User: {userId}
            </p>
            {/* Persona Summary */}
            {loadingPersona ? (
              <div className="text-xs text-gray-400 italic">Analyzing your reading profile...</div>
            ) : persona?.summary ? (
              <div className="bg-[#faf9f6] border-l-4 border-[#b392ac] p-4">
                <p className="text-sm text-[#555] leading-relaxed italic">{persona.summary}</p>
              </div>
            ) : (
              <div className="bg-[#faf9f6] border-l-4 border-gray-200 p-4">
                <p className="text-xs text-gray-400 italic">
                  Add more books to your collection to generate a reading persona.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-[#eee] p-5 text-center group hover:border-[#b392ac] transition-colors">
          <BookOpen className="w-5 h-5 text-[#b392ac] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#b392ac]">{readingStats.total}</div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mt-1">Total Books</div>
        </div>
        <div className="bg-white border border-[#eee] p-5 text-center group hover:border-[#f4acb7] transition-colors">
          <Target className="w-5 h-5 text-[#f4acb7] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#f4acb7]">{completionRate}%</div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mt-1">Completion Rate</div>
        </div>
        <div className="bg-white border border-[#eee] p-5 text-center group hover:border-[#9d7799] transition-colors">
          <Star className="w-5 h-5 text-[#9d7799] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#9d7799]">{avgRating}</div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mt-1">Avg Rating</div>
        </div>
        <div className="bg-white border border-[#eee] p-5 text-center group hover:border-[#735d78] transition-colors">
          <TrendingUp className="w-5 h-5 text-[#735d78] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#735d78]">{readingStats.reading}</div>
          <div className="text-[10px] text-gray-400 uppercase tracking-wider mt-1">Currently Reading</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Favorite Authors & Genres */}
        <div className="bg-white border border-[#eee] p-6 shadow-sm">
          <h3 className="text-xs font-bold uppercase tracking-widest text-[#b392ac] mb-4 flex items-center gap-2">
            <Award className="w-4 h-4" /> Favorite Authors
          </h3>
          {persona?.top_authors && persona.top_authors.length > 0 ? (
            <div className="space-y-2">
              {persona.top_authors.slice(0, 5).map((author, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 border border-[#f5f5f5] hover:bg-[#faf9f6] transition-colors"
                >
                  <span className="text-[10px] font-bold text-[#b392ac] w-5">#{idx + 1}</span>
                  <span className="text-sm text-[#555]">{author}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">
              Not enough data yet. Add more books!
            </p>
          )}
        </div>

        <div className="bg-white border border-[#eee] p-6 shadow-sm">
          <h3 className="text-xs font-bold uppercase tracking-widest text-[#b392ac] mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Top Categories
          </h3>
          {persona?.top_categories && persona.top_categories.length > 0 ? (
            <div className="space-y-2">
              {persona.top_categories.slice(0, 5).map((cat, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 border border-[#f5f5f5] hover:bg-[#faf9f6] transition-colors"
                >
                  <span className="text-[10px] font-bold text-[#9d7799] w-5">#{idx + 1}</span>
                  <span className="text-sm text-[#555]">{cat}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">
              Not enough data yet. Add more books!
            </p>
          )}
        </div>
      </div>

      {/* Rating Distribution */}
      <div className="bg-white border border-[#eee] p-6 shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#b392ac] mb-4 flex items-center gap-2">
          <Star className="w-4 h-4" /> Rating Distribution
        </h3>
        <div className="space-y-3">
          {ratingDistribution.reverse().map(({ star, count }) => (
            <div key={star} className="flex items-center gap-3">
              <div className="flex gap-0.5 w-20 justify-end">
                {[1, 2, 3, 4, 5].map((s) => (
                  <Star
                    key={s}
                    className={`w-3 h-3 ${s <= star ? "text-[#f4acb7] fill-current" : "text-gray-200"
                      }`}
                  />
                ))}
              </div>
              <div className="flex-1 bg-gray-100 h-4 relative overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#f4acb7] to-[#b392ac] transition-all duration-500"
                  style={{ width: `${(count / maxRatingCount) * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-bold text-gray-400 w-6 text-right">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Completion Progress */}
      <div className="bg-white border border-[#eee] p-6 shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#b392ac] mb-4 flex items-center gap-2">
          <Target className="w-4 h-4" /> Reading Progress
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between text-[10px] text-gray-400 uppercase tracking-wider">
            <span>Want to Read ({readingStats.want_to_read})</span>
            <span>Reading ({readingStats.reading})</span>
            <span>Finished ({readingStats.finished})</span>
          </div>
          <div className="h-6 bg-gray-100 flex overflow-hidden">
            {readingStats.total > 0 && (
              <>
                <div
                  className="bg-[#f4acb7] h-full transition-all duration-500 flex items-center justify-center"
                  style={{ width: `${(readingStats.want_to_read / readingStats.total) * 100}%` }}
                >
                  {readingStats.want_to_read > 0 && (
                    <span className="text-[8px] text-white font-bold">
                      {Math.round((readingStats.want_to_read / readingStats.total) * 100)}%
                    </span>
                  )}
                </div>
                <div
                  className="bg-[#9d7799] h-full transition-all duration-500 flex items-center justify-center"
                  style={{ width: `${(readingStats.reading / readingStats.total) * 100}%` }}
                >
                  {readingStats.reading > 0 && (
                    <span className="text-[8px] text-white font-bold">
                      {Math.round((readingStats.reading / readingStats.total) * 100)}%
                    </span>
                  )}
                </div>
                <div
                  className="bg-[#735d78] h-full transition-all duration-500 flex items-center justify-center"
                  style={{ width: `${(readingStats.finished / readingStats.total) * 100}%` }}
                >
                  {readingStats.finished > 0 && (
                    <span className="text-[8px] text-white font-bold">
                      {Math.round((readingStats.finished / readingStats.total) * 100)}%
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Recently Finished */}
      <div className="bg-white border border-[#eee] p-6 shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#b392ac] mb-4 flex items-center gap-2">
          <Clock className="w-4 h-4" /> Recently Finished
        </h3>
        {recentlyFinished.length > 0 ? (
          <div className="grid grid-cols-5 gap-4">
            {recentlyFinished.map((book, idx) => (
              <div key={book.isbn || idx} className="text-center">
                <div className="border border-[#eee] p-1 bg-white shadow-sm mb-2">
                  <img
                    src={book.img || book.thumbnail || PLACEHOLDER_IMG}
                    alt={book.title}
                    className="w-full aspect-[3/4] object-cover"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = PLACEHOLDER_IMG;
                    }}
                  />
                </div>
                <p className="text-[10px] font-bold text-[#555] truncate" title={book.title}>
                  {book.title}
                </p>
                {book.rating > 0 && (
                  <div className="flex justify-center gap-0.5 mt-1">
                    {[1, 2, 3, 4, 5].map((s) => (
                      <Star
                        key={s}
                        className={`w-2 h-2 ${s <= book.rating ? "text-[#f4acb7] fill-current" : "text-gray-200"
                          }`}
                      />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-400 italic text-center py-8">
            No finished books yet. Keep reading!
          </p>
        )}
      </div>
    </div>
  );
};

export default ProfilePage;
