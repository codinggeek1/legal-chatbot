from fastapi import FastAPI
from pydantic import BaseModel
from query_agent import QueryAgent
from summarization_agent import SummarizationAgent

app = FastAPI()

# Load Query and Summarization Agents
query_agent = QueryAgent(pdf_folder="legal_pdfs")
summarization_agent = SummarizationAgent()

class UserQuery(BaseModel):
    question: str

@app.post("/legal-query/")
async def get_legal_response(user_query: UserQuery):
    relevant_text = query_agent.retrieve_relevant_text(user_query.question)
    summarized_response = summarization_agent.summarize_legal_text(relevant_text)
    
    return {"response": summarized_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
