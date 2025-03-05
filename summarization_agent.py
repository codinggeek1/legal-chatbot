from transformers import pipeline

class SummarizationAgent:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_legal_text(self, text):
        """Summarizes legal text into simpler language."""
        summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']

# Initialize summarization agent
summarization_agent = SummarizationAgent()
