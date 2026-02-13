import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Bookmark, User, PlusCircle, Settings, BookOpen, UserCircle } from "lucide-react";

const Header = ({ userId, onUserIdChange, onAddBookClick, onSettingsClick }) => {
  const location = useLocation();

  const navLinks = [
    { path: "/", label: "Gallery", icon: BookOpen },
    { path: "/bookshelf", label: "My Bookshelf", icon: Bookmark },
    { path: "/profile", label: "Profile", icon: UserCircle },
  ];

  return (
    <header className="max-w-5xl mx-auto pt-10 px-4 flex justify-between items-end mb-12">
      <div>
        <Link to="/">
          <div className="border border-[#d0dcc2] px-5 py-2 bg-surface inline-block mb-2 transition-shadow rounded-full shadow-soft">
            <h1 className="text-2xl font-serif font-semibold tracking-[0.08em] text-text-primary">Book Shelf</h1>
          </div>
        </Link>
        <p className="text-[11px] text-text-secondary font-medium tracking-wide">Discover books that resonate with your soul</p>
      </div>
      <div className="flex gap-2 items-center bg-surface border border-[#d0dcc2] rounded-full px-3 py-2 shadow-soft">
        {/* User Switcher */}
        <div className="flex items-center gap-2 border border-[#d8e4cd] bg-surface px-3 py-2 rounded-full" title="Switch User">
          <User className="w-3 h-3 text-text-secondary" />
          <input
            className="w-20 text-[11px] outline-none text-text-primary font-medium bg-transparent placeholder-text-secondary"
            value={userId}
            onChange={(e) => onUserIdChange(e.target.value)}
            placeholder="User ID"
          />
        </div>

        {/* Add Book Button */}
        <button
          onClick={onAddBookClick}
          className="flex items-center gap-1 px-4 py-2 bg-text-primary border border-text-primary transition-all text-[11px] text-surface font-semibold mr-1 group rounded-full"
        >
          <PlusCircle className="w-3 h-3 text-surface" /> Add Book
        </button>

        {/* Navigation Links */}
        {navLinks.map(({ path, label, icon: Icon }) => (
          <Link
            key={path}
            to={path}
            className={`px-4 py-2 text-sm font-medium transition-all flex items-center gap-1 rounded-full ${
              location.pathname === path
                ? "bg-[#F0FDF4] text-text-primary border border-[#d6e4ce]"
                : "bg-transparent text-text-secondary border border-transparent hover:border-[#d6e4ce] hover:bg-[#F0FDF4]"
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </Link>
        ))}

        {/* Settings */}
        <button
          onClick={onSettingsClick}
          className="p-2 hover:bg-[#F0FDF4] rounded-full transition-colors"
          title="Settings"
        >
          <Settings className="w-4 h-4 text-text-secondary" />
        </button>
      </div>
    </header>
  );
};

export default Header;
