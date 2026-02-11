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
          <div className="border border-[#333] px-4 py-1 bg-white shadow-[2px_2px_0px_0px_#eee] inline-block mb-2 hover:shadow-[3px_3px_0px_0px_#ddd] transition-shadow">
            <h1 className="text-xl font-bold uppercase tracking-[0.2em] text-[#333]">Paper Shelf</h1>
          </div>
        </Link>
        <p className="text-[10px] text-gray-400 font-medium tracking-widest">Discover books that resonate with your soul</p>
      </div>
      <div className="flex gap-2 items-center">
        {/* User Switcher */}
        <div className="flex items-center gap-2 border border-[#eee] bg-white px-2 py-1 shadow-sm mr-2" title="Switch User">
          <User className="w-3 h-3 text-gray-400" />
          <input
            className="w-20 text-[10px] outline-none text-gray-600 font-bold bg-transparent placeholder-gray-300"
            value={userId}
            onChange={(e) => onUserIdChange(e.target.value)}
            placeholder="User ID"
          />
        </div>

        {/* Add Book Button */}
        <button
          onClick={onAddBookClick}
          className="flex items-center gap-1 px-3 py-1 bg-white border border-[#333] shadow-sm hover:shadow-md transition-all text-[10px] font-bold uppercase tracking-widest mr-2 group"
        >
          <PlusCircle className="w-3 h-3 text-[#b392ac] group-hover:text-[#9d7799]" /> Add Book
        </button>

        {/* Navigation Links */}
        {navLinks.map(({ path, label, icon: Icon }) => (
          <Link
            key={path}
            to={path}
            className={`px-4 py-2 text-sm font-bold transition-all flex items-center gap-1 ${
              location.pathname === path
                ? "bg-[#b392ac] text-white hover:bg-[#9d7799]"
                : "bg-transparent text-[#b392ac] border-b-2 border-transparent hover:border-[#b392ac]"
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </Link>
        ))}

        {/* Settings */}
        <button
          onClick={onSettingsClick}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          title="Settings"
        >
          <Settings className="w-4 h-4 text-gray-500" />
        </button>
      </div>
    </header>
  );
};

export default Header;
