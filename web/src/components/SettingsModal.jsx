import React from "react";
import { X } from "lucide-react";

const SettingsModal = ({ onClose, apiKey, onApiKeyChange, llmProvider, onProviderChange, onSave }) => {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/10 backdrop-blur-sm animate-in fade-in">
      <div className="bg-white p-6 shadow-xl border border-[#333] w-full max-w-md relative">
        <button onClick={onClose} className="absolute top-2 right-2">
          <X className="w-4 h-4" />
        </button>
        <h3 className="font-bold uppercase tracking-widest mb-4 text-[#b392ac]">Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-bold text-gray-500 mb-1">LLM Provider</label>
            <select
              value={llmProvider}
              onChange={(e) => onProviderChange(e.target.value)}
              className="w-full border p-2 text-sm outline-none focus:border-[#b392ac] bg-white"
            >
              <option value="openai">OpenAI (Requires Key)</option>
              <option value="ollama">Ollama (Local Default)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold text-gray-500 mb-1">OpenAI API Key</label>
            <input
              type="password"
              className="w-full border p-2 text-sm outline-none focus:border-[#b392ac]"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
            />
            <p className="text-[9px] text-gray-400 mt-1">
              Required if using OpenAI. For Ollama/Mock, this is ignored. Stored locally.
            </p>
          </div>
          <button
            onClick={onSave}
            className="w-full px-4 py-2 text-sm font-bold transition-all bg-[#b392ac] text-white hover:bg-[#9d7799]"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
