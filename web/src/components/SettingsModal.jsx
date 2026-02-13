import React from "react";
import { X } from "lucide-react";

const SettingsModal = ({ onClose, apiKey, onApiKeyChange, llmProvider, onProviderChange, onSave }) => {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/20 animate-in fade-in">
      <div className="bg-white p-6 shadow-soft border border-[#d3dec7] rounded-2xl w-full max-w-md relative">
        <button onClick={onClose} className="absolute top-2 right-2">
          <X className="w-4 h-4" />
        </button>
        <h3 className="font-bold uppercase tracking-widest mb-4 text-[#5E81AC]">Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-bold text-[#4C566A] mb-1">LLM Provider</label>
            <select
              value={llmProvider}
              onChange={(e) => onProviderChange(e.target.value)}
              className="w-full border border-info2 p-2 text-sm bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-md"
            >
              <option value="openai">OpenAI (GPT-3.5/4)</option>
              <option value="groq">Groq (Llama3 - Free/Fast)</option>
              <option value="deepseek">DeepSeek (DeepSeek Chat)</option>
              <option value="ollama">Ollama (Local)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold text-[#4C566A] mb-1">API Key (OpenAI / Groq / DeepSeek)</label>
            <input
              type="password"
              className="w-full border border-info2 p-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info/30 focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-md"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
            />
            <p className="text-[9px] text-[#81A1C1] mt-1">
              Required for OpenAI, Groq, or DeepSeek. Stored locally.
            </p>
          </div>
          <button
            onClick={onSave}
            className="w-full px-4 py-2 text-sm font-bold transition-all bg-[#5E81AC] text-white hover:bg-[#4C566A]"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
