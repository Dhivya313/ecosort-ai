"""
EcoSort AI — Waste Analyzer Module
Handles waste identification and classification using the Gemini API.
"""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_CATEGORIES = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Organic",
    "E-Waste",
    "Other",
]

SUPPORTED_MEDIA_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
]

ANALYSIS_PROMPT = """\
You are an expert waste classification assistant. Analyze the provided image \
and identify the primary visible waste item.

Classify the item into exactly ONE of these categories:
- Plastic
- Paper
- Glass
- Metal
- Organic
- E-Waste
- Other

Respond with ONLY valid JSON in this exact format — no markdown fences, \
no extra text, no commentary:
{
  "item": "brief description of the identified waste item",
  "category": "one of the valid categories listed above",
  "confidence": <integer from 0 to 100>,
  "reasoning": "brief explanation of why you classified it this way"
}

Rules:
- If the image is unclear or blurry, still return valid JSON but set \
confidence to a low value and explain in reasoning.
- If no waste is visible, use category "Other" with low confidence.
- confidence must be an integer between 0 and 100.
"""


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class WasteAnalysisError(Exception):
    """Base exception for waste analysis errors."""
    pass


class MissingAPIKeyError(WasteAnalysisError):
    """Raised when the Gemini API key is not configured."""
    pass


class UnsupportedImageError(WasteAnalysisError):
    """Raised when the image format is not supported."""
    pass


class InvalidResponseError(WasteAnalysisError):
    """Raised when Gemini returns invalid or unparseable JSON."""
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_response(data: dict) -> dict:
    """Validate and normalise the parsed JSON response from Gemini.

    Returns the validated (and possibly coerced) dict.
    Raises InvalidResponseError on any validation failure.
    """
    required_fields = ["item", "category", "confidence", "reasoning"]

    for field in required_fields:
        if field not in data:
            raise InvalidResponseError(f"Missing required field: '{field}'")

    # --- item ---
    if not isinstance(data["item"], str) or not data["item"].strip():
        raise InvalidResponseError("'item' must be a non-empty string")

    # --- category ---
    if data["category"] not in VALID_CATEGORIES:
        raise InvalidResponseError(
            f"Invalid category '{data['category']}'. "
            f"Must be one of: {VALID_CATEGORIES}"
        )

    # --- confidence ---
    try:
        data["confidence"] = int(data["confidence"])
    except (TypeError, ValueError):
        raise InvalidResponseError("'confidence' must be an integer")

    if not 0 <= data["confidence"] <= 100:
        raise InvalidResponseError(
            f"'confidence' must be between 0 and 100, got {data['confidence']}"
        )

    # --- reasoning ---
    if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
        raise InvalidResponseError("'reasoning' must be a non-empty string")

    return data


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------
def analyze_waste_image(image_bytes: bytes, media_type: str) -> dict:
    """Analyze an image of waste using the Gemini API.

    Args:
        image_bytes: Raw bytes of the image file.
        media_type:  MIME type (e.g. ``"image/jpeg"``, ``"image/png"``).

    Returns:
        dict with keys ``item``, ``category``, ``confidence``, ``reasoning``.

    Raises:
        MissingAPIKeyError:  GEMINI_API_KEY is not set.
        UnsupportedImageError: The media type is not supported or data is empty.
        InvalidResponseError: Gemini returned unparseable / invalid JSON.
        WasteAnalysisError:  Any other API or processing error.
    """
    # ---- Pre-flight checks ------------------------------------------------
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "your_api_key_here":
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedImageError(
            f"Unsupported image type '{media_type}'. "
            f"Supported: {SUPPORTED_MEDIA_TYPES}"
        )

    if not image_bytes:
        raise UnsupportedImageError("Image data is empty.")

    # ---- Call Gemini ------------------------------------------------------
    try:
        client = genai.Client(api_key=api_key)

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=media_type)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[ANALYSIS_PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500,
            ),
        )

        raw_text = response.text.strip()

    except (MissingAPIKeyError, UnsupportedImageError):
        raise
    except Exception as exc:
        raise WasteAnalysisError(f"Gemini API error: {exc}") from exc

    # ---- Parse & validate -------------------------------------------------
    # Strip markdown code fences if Gemini wrapped the output
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise InvalidResponseError(
            f"Failed to parse Gemini response as JSON: {exc}\nRaw: {raw_text}"
        ) from exc

    return _validate_response(data)
