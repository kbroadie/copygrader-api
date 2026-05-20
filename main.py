import os
import json
import anthropic
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="CopyLint API",
    description="Analyze marketing copy for vague promises, fluffery, and clickbait. Built by a 25-year marketing professional.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a senior marketing copy analyst with 25 years of experience in brand communications and advertising. Your job is to objectively score and critique marketing copy.

You flag three core problems:
1. VAGUE PROMISES — language that implies a benefit without naming one (e.g. "take your business to the next level", "solutions that work for you", "transform your world")
2. FLUFFERY — empty adjective pileups and meaningless superlatives (e.g. "world-class", "best-in-class", "revolutionary", "cutting-edge", "game-changing", "innovative")  
3. CLICKBAIT — manufactured urgency, sensationalism, or withholding to bait a click (e.g. "you won't believe", "this one trick", "what they don't want you to know", "limited time" with no real deadline)

You also note:
- JARGON — buzzwords that obscure rather than clarify ("synergize", "leverage", "paradigm shift", "ecosystem", "bandwidth" in non-technical contexts)
- READING LEVEL — approximate US grade level of the text

Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

Return this exact structure:
{
  "overall_score": <integer 0-100, where 100 is excellent clean copy>,
  "grade": <"A", "B", "C", "D", or "F">,
  "verdict": <one blunt sentence summary of the copy's biggest problem, or praise if it's good>,
  "flags": {
    "vague_promises": <list of exact phrases from the input that are vague promises, empty list if none>,
    "fluffery": <list of exact phrases that are fluffery, empty list if none>,
    "clickbait": <list of exact phrases that are clickbait, empty list if none>,
    "jargon": <list of jargon words/phrases, empty list if none>
  },
  "flag_count": <total number of flagged phrases across all categories>,
  "reading_level": <"Grade X" format, e.g. "Grade 8">,
  "char_count": <integer character count of input>,
  "word_count": <integer word count of input>,
  "platform_fit": {
    "google_ad_headline": <"fits (X chars)" or "too long (30 char limit)">,
    "facebook_ad": <"fits" or "too long (125 char primary text limit)">,
    "twitter_x": <"fits (X chars)" or "too long (280 char limit)">,
    "email_subject": <"fits" or "too long (60 char recommended)">
  },
  "rewrite": <a cleaner rewrite of the copy that fixes the flagged issues, or null if the copy is already good (score >= 80)>
}"""


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="The marketing copy to analyze (max 5000 characters)")
    include_rewrite: Optional[bool] = Field(True, description="Whether to include a rewritten version. Set to false to save tokens on bulk analysis.")


@app.get("/")
def root():
    return {
        "api": "CopyLint",
        "version": "1.0.0",
        "description": "Analyze marketing copy for vague promises, fluffery, and clickbait",
        "endpoints": {
            "POST /analyze": "Analyze a piece of marketing copy",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_copy(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    user_prompt = f"Analyze this marketing copy:\n\n{request.text}"
    if not request.include_rewrite:
        user_prompt += "\n\nSet 'rewrite' to null in your response regardless of score."

    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        raw = message.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse analysis result. Please try again.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
