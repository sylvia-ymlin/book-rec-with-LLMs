import { useState } from "react";

export function useLlmSettings() {
  const [apiKey, setApiKey] = useState(
    () => localStorage.getItem("openai_key") || ""
  );
  const [llmProvider, setLlmProvider] = useState(() => {
    const stored = localStorage.getItem("llm_provider");
    const allowed = new Set(["ollama", "openai", "groq", "deepseek"]);
    if (!stored || stored === "mock" || !allowed.has(stored)) {
      return "ollama";
    }
    return stored;
  });

  const saveSettings = () => {
    localStorage.setItem("openai_key", apiKey);
    localStorage.setItem("llm_provider", llmProvider);
  };

  return {
    apiKey,
    setApiKey,
    llmProvider,
    setLlmProvider,
    saveSettings,
  };
}

