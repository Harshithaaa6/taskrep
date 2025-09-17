# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GuessRequest(BaseModel):
    guess: str
    correct: str

def random_color():
    return f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})"

@app.get("/start")
def start_game():
    correct = random_color()
    options = [correct] + [random_color() for _ in range(3)]
    random.shuffle(options)
    return {"correct": correct, "options": options}

@app.post("/check")
def check_guess(request: GuessRequest):
    if request.guess == request.correct:
        return {"message": "✅ Correct! You guessed the color!"}
    return {"message": "❌ Wrong guess, try again!"}
