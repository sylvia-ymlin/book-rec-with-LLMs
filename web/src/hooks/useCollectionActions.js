import {
  addFavorite,
  removeFromFavorites,
  updateBook,
  getFavorites,
  getUserStats,
} from "../api";

export function useCollectionActions({ userId, myCollection, setMyCollection, setReadingStats }) {
  const refreshCollection = async () => {
    const [favs, stats] = await Promise.all([
      getFavorites(userId),
      getUserStats(userId),
    ]);
    setMyCollection(favs);
    setReadingStats(stats);
  };

  const toggleCollect = async (book) => {
    try {
      if (myCollection.some((b) => b.isbn === book.isbn)) {
        await removeFromFavorites(book.isbn, userId);
      } else {
        await addFavorite(book.isbn, userId);
      }
      await refreshCollection();
    } catch (err) {
      console.error(err);
    }
  };

  const handleRatingChange = async (isbn, rating) => {
    try {
      await updateBook(isbn, { rating }, userId);
      setMyCollection((prev) =>
        prev.map((book) => (book.isbn === isbn ? { ...book, rating } : book))
      );
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleStatusChange = async (isbn, status) => {
    try {
      await updateBook(isbn, { status }, userId);
      setMyCollection((prev) =>
        prev.map((book) => (book.isbn === isbn ? { ...book, status } : book))
      );
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleRemoveBook = async (isbn) => {
    try {
      await removeFromFavorites(isbn, userId);
      setMyCollection((prev) => prev.filter((book) => book.isbn !== isbn));
      getUserStats(userId)
        .then((stats) => setReadingStats(stats))
        .catch(console.error);
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpdateComment = (isbn, value, persist) => {
    setMyCollection((prev) =>
      prev.map((b) => (b.isbn === isbn ? { ...b, comment: value } : b))
    );
    if (persist) {
      updateBook(isbn, { comment: value }, userId).catch(console.error);
    }
  };

  return {
    refreshCollection,
    toggleCollect,
    handleRatingChange,
    handleStatusChange,
    handleRemoveBook,
    handleUpdateComment,
  };
}

