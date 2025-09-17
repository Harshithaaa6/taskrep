from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="AI Day Designer")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a text generation model (small & fast for demo)
generator = pipeline("text-generation", model="gpt2")

# Input format
class DayRequest(BaseModel):
    morning: str
    afternoon: str
    evening: str

@app.post("/design_day")
def design_day(request: DayRequest):
    user_input = (
        f"Plan an ideal productive and balanced day based on these preferences:\n"
        f"- Morning: {request.morning}\n"
        f"- Afternoon: {request.afternoon}\n"
        f"- Evening: {request.evening}\n"
        f"Provide a short, positive schedule."
    )

    response = generator(user_input, max_length=100, num_return_sequences=1)
    text = response[0]["generated_text"]

    # For simplicity, just return user activities with some AI flavor
    plan = {
        "morning": f"{request.morning} + energizing start (AI suggests meditation/workout)",
        "afternoon": f"{request.afternoon} + focused session (AI suggests short breaks)",
        "evening": f"{request.evening} + relaxing close (AI suggests journaling/reading)"
    }

    return {"plan": plan, "raw_ai": text}
