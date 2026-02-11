from typing import List, Dict

class DialogueManager:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_turn(self, user_input: str, system_response: str):
        """
        Adds a single turn to the history.
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": system_response})
        
        # Keep history within limit (rolling buffer)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns the conversation history.
        """
        return self.history

    def clear_history(self):
        """
        Resets the conversation.
        """
        self.history = []

    def get_context_string(self) -> str:
        """
        Returns history formatted as a string for simple prompts.
        """
        context = ""
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Agent"
            context += f"{role}: {turn['content']}\n"
        return context
