import React, { useState, useEffect } from "react";
import { BookOpen, Star, Target, TrendingUp, Clock, Award, BarChart3 } from "lucide-react";
import { getPersona } from "../api";
import { PLACEHOLDER_IMG } from "../constants";

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
      <div className="bg-white border border-[#d0dcc2] p-8 rounded-[32px] shadow-soft">
        <div className="flex items-start gap-6">
          <div className="w-20 h-20 rounded-full overflow-hidden border border-[#dbe6d2] shadow-soft bg-white">
            <img
              src="/content/avatar-cat.png"
              alt="User avatar"
              className="w-full h-full object-cover"
            />
          </div>
          <div className="flex-1">
            <h2 className="text-3xl font-serif font-semibold text-text-primary mb-1">Reader Profile</h2>
            <p className="text-xs text-text-secondary font-semibold uppercase tracking-widest mb-4">
              User: {userId}
            </p>
            {/* Persona Summary */}
            {loadingPersona ? (
              <div className="text-xs text-gray-400 italic">Analyzing your reading profile...</div>
            ) : persona?.summary ? (
              <div className="bg-[#F0FDF4] border-l-4 border-[#5E81AC] p-4 rounded-r-2xl">
                <p className="text-sm text-text-primary leading-relaxed italic">{persona.summary}</p>
              </div>
            ) : (
              <div className="bg-[#F0FDF4] border-l-4 border-[#88C0D0] p-4 rounded-r-2xl">
                <p className="text-xs text-text-secondary italic">
                  Add more books to your collection to generate a reading persona.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-[#d3dfc8] p-5 text-center group hover:border-[#5E81AC] transition-colors rounded-3xl shadow-soft">
          <BookOpen className="w-5 h-5 text-[#5E81AC] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#5E81AC]">{readingStats.total}</div>
          <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider mt-1">Total Books</div>
        </div>
        <div className="bg-white border border-[#E5C49D] p-5 text-center group hover:border-[#D08770] transition-colors rounded-3xl shadow-soft">
          <Target className="w-5 h-5 text-[#D08770] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#D08770]">{completionRate}%</div>
          <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider mt-1">Completion Rate</div>
        </div>
        <div className="bg-white border border-[#c8d9d4] p-5 text-center group hover:border-[#81A1C1] transition-colors rounded-3xl shadow-soft">
          <Star className="w-5 h-5 text-[#81A1C1] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#81A1C1]">{avgRating}</div>
          <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider mt-1">Avg Rating</div>
        </div>
        <div className="bg-white border border-[#d3dfc8] p-5 text-center group hover:border-[#4C566A] transition-colors rounded-3xl shadow-soft">
          <TrendingUp className="w-5 h-5 text-[#4C566A] mx-auto mb-2" />
          <div className="text-3xl font-bold text-[#4C566A]">{readingStats.reading}</div>
          <div className="text-[10px] text-[#81A1C1] uppercase tracking-wider mt-1">Currently Reading</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Favorite Authors & Genres */}
        <div className="bg-white border border-[#d3dfc8] p-6 rounded-[32px] shadow-soft">
          <h3 className="text-xs font-bold uppercase tracking-widest text-[#5E81AC] mb-4 flex items-center gap-2">
            <Award className="w-4 h-4" /> Favorite Authors
          </h3>
          {persona?.top_authors && persona.top_authors.length > 0 ? (
            <div className="space-y-2">
              {persona.top_authors.slice(0, 5).map((author, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 border border-[#dbe6d2] rounded-2xl hover:bg-[#F0FDF4] transition-colors"
                >
                  <span className="text-[10px] font-bold text-[#5E81AC] w-5">#{idx + 1}</span>
                  <span className="text-sm text-[#4C566A]">{author}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">
              Not enough data yet. Add more books!
            </p>
          )}
        </div>

        <div className="bg-white border border-[#d3dfc8] p-6 rounded-[32px] shadow-soft">
          <h3 className="text-xs font-bold uppercase tracking-widest text-[#5E81AC] mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Top Categories
          </h3>
          {persona?.top_categories && persona.top_categories.length > 0 ? (
            <div className="space-y-2">
              {persona.top_categories.slice(0, 5).map((cat, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-2 border border-[#dbe6d2] rounded-2xl hover:bg-[#F0FDF4] transition-colors"
                >
                  <span className="text-[10px] font-bold text-[#81A1C1] w-5">#{idx + 1}</span>
                  <span className="text-sm text-[#4C566A]">{cat}</span>
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
      <div className="bg-white border border-[#d3dfc8] p-6 rounded-[32px] shadow-soft">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#5E81AC] mb-4 flex items-center gap-2">
          <Star className="w-4 h-4" /> Rating Distribution
        </h3>
        <div className="space-y-3">
          {ratingDistribution.reverse().map(({ star, count }) => (
            <div key={star} className="flex items-center gap-3">
              <div className="flex gap-0.5 w-20 justify-end">
                {[1, 2, 3, 4, 5].map((s) => (
                  <Star
                    key={s}
                    className={`w-3 h-3 ${s <= star ? "text-[#D08770] fill-current" : "text-gray-200"}`}
                  />
                ))}
              </div>
              <div className="flex-1 bg-gray-100 h-4 relative overflow-hidden rounded-full">
                <div
                  className="h-full bg-[#D08770] transition-all duration-500 rounded-full"
                  style={{ width: `${(count / maxRatingCount) * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-bold text-gray-400 w-6 text-right">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Completion Progress */}
      <div className="bg-white border border-[#d3dfc8] p-6 rounded-[32px] shadow-soft">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#5E81AC] mb-4 flex items-center gap-2">
          <Target className="w-4 h-4" /> Reading Progress
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between text-[10px] text-gray-400 uppercase tracking-wider">
            <span>Want to Read ({readingStats.want_to_read})</span>
            <span>Reading ({readingStats.reading})</span>
            <span>Finished ({readingStats.finished})</span>
          </div>
          <div className="h-6 bg-gray-100 flex overflow-hidden rounded-full">
            {readingStats.total > 0 && (
              <>
                <div
                  className="bg-[#D08770] h-full transition-all duration-500 flex items-center justify-center"
                  style={{ width: `${(readingStats.want_to_read / readingStats.total) * 100}%` }}
                >
                  {readingStats.want_to_read > 0 && (
                    <span className="text-[8px] text-white font-bold">
                      {Math.round((readingStats.want_to_read / readingStats.total) * 100)}%
                    </span>
                  )}
                </div>
                <div
                  className="bg-[#81A1C1] h-full transition-all duration-500 flex items-center justify-center"
                  style={{ width: `${(readingStats.reading / readingStats.total) * 100}%` }}
                >
                  {readingStats.reading > 0 && (
                    <span className="text-[8px] text-white font-bold">
                      {Math.round((readingStats.reading / readingStats.total) * 100)}%
                    </span>
                  )}
                </div>
                <div
                  className="bg-[#5E81AC] h-full transition-all duration-500 flex items-center justify-center"
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
      <div className="bg-white border border-[#d3dfc8] p-6 rounded-[32px] shadow-soft">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[#5E81AC] mb-4 flex items-center gap-2">
          <Clock className="w-4 h-4" /> Recently Finished
        </h3>
        {recentlyFinished.length > 0 ? (
          <div className="grid grid-cols-5 gap-4">
            {recentlyFinished.map((book, idx) => (
              <div key={book.isbn || idx} className="text-center">
                <div className="border border-[#d3dfc8] p-1 bg-white shadow-soft mb-2 rounded-2xl">
                  <img
                    src={book.img || book.thumbnail || PLACEHOLDER_IMG}
                    alt={book.title}
                    className="w-full aspect-[3/4] object-cover rounded-xl"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = PLACEHOLDER_IMG;
                    }}
                  />
                </div>
                <p className="text-[10px] font-bold text-[#4C566A] truncate" title={book.title}>
                  {book.title}
                </p>
                {book.rating > 0 && (
                  <div className="flex justify-center gap-0.5 mt-1">
                    {[1, 2, 3, 4, 5].map((s) => (
                      <Star
                        key={s}
                        className={`w-2 h-2 ${s <= book.rating ? "text-[#D08770] fill-current" : "text-gray-200"}`}
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
