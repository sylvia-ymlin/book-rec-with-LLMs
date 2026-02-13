import { useState } from "react";

const MAX_RECENT_ISBNS = 10;

export function useSessionRecents() {
  const [recentIsbns, setRecentIsbns] = useState([]);

  const trackRecentIsbn = (isbn) => {
    if (!isbn) return;
    setRecentIsbns((prev) =>
      [isbn, ...prev.filter((item) => item !== isbn)].slice(0, MAX_RECENT_ISBNS)
    );
  };

  return {
    recentIsbns,
    trackRecentIsbn,
  };
}

