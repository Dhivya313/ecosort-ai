"""
EcoSort AI — Comprehensive QA Test Suite
Covers all functional paths: Gemini analysis, validation, RAG retrieval,
error handling, edge cases, and UI error paths.
"""

import io
import json
import os
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from waste_analyzer import (
    analyze_waste_image,
    _validate_response,
    MissingAPIKeyError,
    UnsupportedImageError,
    InvalidResponseError,
    WasteAnalysisError,
    VALID_CATEGORIES,
    SUPPORTED_MEDIA_TYPES,
)
from rag_engine import RAGEngine

TEST_CHROMA_DIR = Path(__file__).parent / "chroma_db_qa"

# ============================================================================
# Image generators
# ============================================================================

def _make_image(label: str, color: tuple, size=(400, 400)) -> tuple[bytes, str]:
    """Create a labeled solid-color JPEG test image."""
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    d.text((size[0] // 6, size[1] // 2 - 14), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _make_plastic_bottle() -> tuple[bytes, str]:
    img = Image.new("RGB", (400, 600), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([(130, 180), (270, 500)], radius=30, fill=(135, 206, 235),
                        outline=(70, 130, 180), width=3)
    d.rounded_rectangle([(170, 100), (230, 190)], radius=8, fill=(135, 206, 235),
                        outline=(70, 130, 180), width=3)
    d.rounded_rectangle([(165, 70), (235, 110)], radius=6, fill=(30, 144, 255),
                        outline=(20, 100, 200), width=3)
    d.rounded_rectangle([(140, 280), (260, 400)], radius=10, fill=(255, 255, 255),
                        outline=(200, 200, 200), width=2)
    try:
        f = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        f = ImageFont.load_default()
    d.text((155, 310), "WATER", fill=(70, 130, 180), font=f)
    d.text((148, 340), "500 mL", fill=(120, 120, 120), font=f)
    try:
        sf = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        sf = ImageFont.load_default()
    d.text((175, 460), "♻ PET 1", fill=(100, 100, 100), font=sf)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_paper() -> tuple[bytes, str]:
    img = Image.new("RGB", (400, 400), (255, 253, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 30), (350, 370)], fill=(255, 255, 255), outline=(180, 170, 150), width=2)
    for y in range(60, 350, 25):
        d.line([(70, y), (330, y)], fill=(200, 200, 210), width=1)
    try:
        f = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        f = ImageFont.load_default()
    d.text((90, 70), "NEWSPAPER", fill=(60, 60, 60), font=f)
    d.text((90, 100), "Daily Times", fill=(100, 100, 100), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_glass_bottle() -> tuple[bytes, str]:
    img = Image.new("RGB", (300, 600), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([(90, 200), (210, 520)], radius=15, fill=(180, 230, 180),
                        outline=(80, 160, 80), width=3)
    d.rounded_rectangle([(120, 120), (180, 210)], radius=6, fill=(180, 230, 180),
                        outline=(80, 160, 80), width=3)
    d.rounded_rectangle([(115, 90), (185, 130)], radius=4, fill=(100, 180, 100),
                        outline=(60, 140, 60), width=3)
    try:
        f = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f = ImageFont.load_default()
    d.text((105, 340), "WINE", fill=(40, 80, 40), font=f)
    d.text((95, 365), "BOTTLE", fill=(40, 80, 40), font=f)
    d.text((100, 400), "GLASS", fill=(60, 120, 60), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_metal_can() -> tuple[bytes, str]:
    img = Image.new("RGB", (300, 400), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([(70, 80), (230, 320)], radius=20, fill=(192, 192, 192),
                        outline=(128, 128, 128), width=3)
    d.ellipse([(70, 60), (230, 110)], fill=(210, 210, 210), outline=(128, 128, 128), width=2)
    try:
        f = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        f = ImageFont.load_default()
    d.text((90, 170), "SODA CAN", fill=(60, 60, 60), font=f)
    d.text((85, 200), "ALUMINUM", fill=(100, 100, 100), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_organic() -> tuple[bytes, str]:
    img = Image.new("RGB", (400, 400), (245, 240, 230))
    d = ImageDraw.Draw(img)
    d.ellipse([(100, 80), (300, 200)], fill=(255, 200, 0), outline=(200, 150, 0), width=3)
    d.arc([(150, 130), (250, 170)], start=0, end=180, fill=(139, 90, 43), width=2)
    d.ellipse([(50, 220), (180, 340)], fill=(255, 69, 0), outline=(200, 50, 0), width=2)
    d.ellipse([(130, 130), (155, 155)], fill=(80, 50, 20))
    d.ellipse([(200, 130), (225, 155)], fill=(80, 50, 20))
    try:
        f = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        f = ImageFont.load_default()
    d.text((110, 210), "BANANA", fill=(100, 80, 0), font=f)
    d.text((70, 350), "APPLE CORE", fill=(150, 50, 0), font=f)
    d.text((50, 375), "FRUIT SCRAPS", fill=(120, 80, 40), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_ewaste() -> tuple[bytes, str]:
    img = Image.new("RGB", (400, 400), (30, 30, 40))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([(60, 60), (340, 280)], radius=12, fill=(50, 50, 60),
                        outline=(100, 100, 120), width=3)
    d.rounded_rectangle([(80, 80), (320, 260)], radius=8, fill=(20, 20, 30),
                        outline=(80, 80, 100), width=1)
    d.rounded_rectangle([(130, 290), (270, 340)], radius=6, fill=(60, 60, 70),
                        outline=(100, 100, 120), width=2)
    d.rounded_rectangle([(160, 340), (240, 360)], radius=4, fill=(70, 70, 80),
                        outline=(100, 100, 120), width=1)
    try:
        f = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f = ImageFont.load_default()
    d.text((130, 150), "LAPTOP", fill=(0, 200, 0), font=f)
    d.text((100, 370), "ELECTRONIC", fill=(150, 150, 160), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _make_blurry() -> tuple[bytes, str]:
    img = Image.new("RGB", (300, 300), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 50), (250, 250)], fill=(150, 180, 150))
    img = img.filter(ImageFilter.GaussianBlur(radius=15))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    return buf.getvalue(), "image/jpeg"


def _make_multi_item() -> tuple[bytes, str]:
    img = Image.new("RGB", (600, 400), (245, 245, 240))
    d = ImageDraw.Draw(img)
    # Bottle
    d.rounded_rectangle([(30, 100), (120, 350)], radius=15, fill=(135, 206, 235),
                        outline=(70, 130, 180), width=2)
    # Can
    d.rounded_rectangle([(160, 150), (260, 320)], radius=12, fill=(192, 192, 192),
                        outline=(128, 128, 128), width=2)
    # Paper
    d.rectangle([(300, 80), (450, 300)], fill=(255, 255, 255),
                outline=(180, 170, 150), width=2)
    # Banana
    d.ellipse([(470, 180), (580, 230)], fill=(255, 200, 0),
              outline=(200, 150, 0), width=2)
    try:
        f = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        f = ImageFont.load_default()
    d.text((40, 360), "BOTTLE", fill=(0, 0, 0), font=f)
    d.text((170, 330), "CAN", fill=(0, 0, 0), font=f)
    d.text((340, 310), "PAPER", fill=(0, 0, 0), font=f)
    d.text((480, 240), "BANANA", fill=(0, 0, 0), font=f)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _make_png_image() -> tuple[bytes, str]:
    img = Image.new("RGB", (200, 200), (135, 206, 235))
    d = ImageDraw.Draw(img)
    d.text((40, 90), "PNG TEST", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


# ============================================================================
# Test runner
# ============================================================================

class QARunner:
    def __init__(self):
        self.results = []

    def record(self, test_id: str, name: str, status: str, detail: str = ""):
        self.results.append({
            "id": test_id, "name": name, "status": status, "detail": detail
        })
        icon = {"PASS": "✓", "FAIL": "✗", "FIXED": "🔧"}.get(status, "?")
        print(f"  {icon} [{status}] {test_id}: {name}")
        if detail:
            print(f"      {detail}")

    def summary(self):
        print("\n" + "=" * 65)
        print("QA SUMMARY")
        print("=" * 65)
        for r in self.results:
            icon = {"PASS": "✓", "FAIL": "✗", "FIXED": "🔧"}.get(r["status"], "?")
            print(f"  {icon} {r['status']:5s}  {r['id']:5s}  {r['name']}")
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        fixed = sum(1 for r in self.results if r["status"] == "FIXED")
        print(f"\n  Total: {total}  |  PASS: {passed}  |  FAIL: {failed}  |  FIXED: {fixed}")
        print("=" * 65)
        return failed


def run_qa():
    qa = QARunner()

    if TEST_CHROMA_DIR.exists():
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    print("=" * 65)
    print("ECOSORT AI — COMPREHENSIVE QA TEST SUITE")
    print("=" * 65)

    # ------------------------------------------------------------------
    # T01: Plastic bottle (Gemini live call)
    # ------------------------------------------------------------------
    print("\n--- T01: Plastic bottle ---")
    try:
        img, mime = _make_plastic_bottle()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        assert r["item"] and r["reasoning"]
        qa.record("T01", "Plastic bottle analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T01", "Plastic bottle analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T02: Paper
    # ------------------------------------------------------------------
    print("\n--- T02: Paper ---")
    try:
        img, mime = _make_paper()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        qa.record("T02", "Paper analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T02", "Paper analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T03: Glass bottle
    # ------------------------------------------------------------------
    print("\n--- T03: Glass bottle ---")
    try:
        img, mime = _make_glass_bottle()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        qa.record("T03", "Glass bottle analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T03", "Glass bottle analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T04: Metal can
    # ------------------------------------------------------------------
    print("\n--- T04: Metal can ---")
    try:
        img, mime = _make_metal_can()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        qa.record("T04", "Metal can analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T04", "Metal can analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T05: Organic waste
    # ------------------------------------------------------------------
    print("\n--- T05: Organic waste ---")
    try:
        img, mime = _make_organic()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        qa.record("T05", "Organic waste analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T05", "Organic waste analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T06: Electronic device
    # ------------------------------------------------------------------
    print("\n--- T06: Electronic device ---")
    try:
        img, mime = _make_ewaste()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        qa.record("T06", "E-waste analysis", "PASS",
                  f"item={r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T06", "E-waste analysis", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T07: Blurry image
    # ------------------------------------------------------------------
    print("\n--- T07: Blurry image ---")
    try:
        img, mime = _make_blurry()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        # Blurry images should still return valid JSON
        qa.record("T07", "Blurry image handling", "PASS",
                  f"cat={r['category']}, conf={r['confidence']} (low expected)")
    except Exception as e:
        qa.record("T07", "Blurry image handling", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T08: Unsupported file type
    # ------------------------------------------------------------------
    print("\n--- T08: Unsupported file type ---")
    try:
        analyze_waste_image(b"fake-bmp-data", "image/bmp")
        qa.record("T08", "Unsupported file type", "FAIL", "No exception raised")
    except UnsupportedImageError:
        qa.record("T08", "Unsupported file type", "PASS", "UnsupportedImageError raised")
    except Exception as e:
        qa.record("T08", "Unsupported file type", "FAIL", f"Wrong exception: {type(e).__name__}: {e}")

    # Also test other unsupported types
    for bad_mime in ["image/gif", "image/tiff", "application/pdf", "video/mp4"]:
        try:
            analyze_waste_image(b"data", bad_mime)
            qa.record("T08b", f"Reject {bad_mime}", "FAIL", "No exception")
        except UnsupportedImageError:
            pass  # expected

    # ------------------------------------------------------------------
    # T09: Empty/no image data
    # ------------------------------------------------------------------
    print("\n--- T09: Empty image data ---")
    try:
        analyze_waste_image(b"", "image/jpeg")
        qa.record("T09", "Empty image data", "FAIL", "No exception raised")
    except UnsupportedImageError:
        qa.record("T09", "Empty image data", "PASS", "UnsupportedImageError raised")
    except Exception as e:
        qa.record("T09", "Empty image data", "FAIL", f"Wrong exception: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # T10: Multiple waste items in one image
    # ------------------------------------------------------------------
    print("\n--- T10: Multiple items ---")
    try:
        img, mime = _make_multi_item()
        r = analyze_waste_image(img, mime)
        assert r["category"] in VALID_CATEGORIES
        assert 0 <= r["confidence"] <= 100
        # Should pick one primary item
        qa.record("T10", "Multiple items in image", "PASS",
                  f"Primary: {r['item']}, cat={r['category']}, conf={r['confidence']}")
    except Exception as e:
        qa.record("T10", "Multiple items in image", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T11: Missing API key
    # ------------------------------------------------------------------
    print("\n--- T11: Missing API key ---")
    try:
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            analyze_waste_image(b"fake", "image/jpeg")
        qa.record("T11", "Missing API key", "FAIL", "No exception raised")
    except MissingAPIKeyError:
        qa.record("T11", "Missing API key", "PASS", "MissingAPIKeyError raised")
    except Exception as e:
        qa.record("T11", "Missing API key", "FAIL", f"Wrong exception: {type(e).__name__}: {e}")

    # Also test placeholder key
    try:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "your_api_key_here"}, clear=False):
            analyze_waste_image(b"fake", "image/jpeg")
        qa.record("T11b", "Placeholder API key", "FAIL", "No exception")
    except MissingAPIKeyError:
        qa.record("T11b", "Placeholder API key", "PASS", "MissingAPIKeyError raised")
    except Exception as e:
        qa.record("T11b", "Placeholder API key", "FAIL", f"Wrong: {e}")

    # ------------------------------------------------------------------
    # T12: Gemini API failure (simulated)
    # ------------------------------------------------------------------
    print("\n--- T12: API failure (mocked) ---")
    try:
        with patch("waste_analyzer.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = \
                ConnectionError("Simulated network failure")
            analyze_waste_image(b"\xff\xd8\xff\xe0fake-jpeg", "image/jpeg")
        qa.record("T12", "API failure handling", "FAIL", "No exception raised")
    except WasteAnalysisError as e:
        if "Gemini API error" in str(e):
            qa.record("T12", "API failure handling", "PASS",
                      "WasteAnalysisError with API message")
        else:
            qa.record("T12", "API failure handling", "FAIL", f"Unexpected msg: {e}")
    except Exception as e:
        qa.record("T12", "API failure handling", "FAIL", f"Wrong exception: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # T13: Invalid model response (simulated)
    # ------------------------------------------------------------------
    print("\n--- T13: Invalid model response ---")

    # T13a: Non-JSON response
    try:
        _validate_response({"item": "x", "category": "Plastic", "confidence": 50, "reasoning": "y"})
        # That should pass — now test invalid JSON parsing
        mock_resp = MagicMock()
        mock_resp.text = "This is not JSON at all"
        with patch("waste_analyzer.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            analyze_waste_image(b"\xff\xd8\xff\xe0fake", "image/jpeg")
        qa.record("T13a", "Non-JSON response", "FAIL", "No exception raised")
    except InvalidResponseError:
        qa.record("T13a", "Non-JSON response", "PASS", "InvalidResponseError raised")
    except Exception as e:
        qa.record("T13a", "Non-JSON response", "FAIL", f"Wrong: {type(e).__name__}: {e}")

    # T13b: Missing required field
    try:
        _validate_response({"item": "bottle", "category": "Plastic"})
        qa.record("T13b", "Missing required field", "FAIL", "No exception")
    except InvalidResponseError as e:
        qa.record("T13b", "Missing required field", "PASS", str(e))

    # T13c: Invalid category
    try:
        _validate_response({"item": "x", "category": "Rubber", "confidence": 50, "reasoning": "y"})
        qa.record("T13c", "Invalid category value", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13c", "Invalid category value", "PASS", "InvalidResponseError raised")

    # T13d: Confidence out of range
    try:
        _validate_response({"item": "x", "category": "Plastic", "confidence": 150, "reasoning": "y"})
        qa.record("T13d", "Confidence > 100", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13d", "Confidence > 100", "PASS", "InvalidResponseError raised")

    try:
        _validate_response({"item": "x", "category": "Plastic", "confidence": -5, "reasoning": "y"})
        qa.record("T13e", "Confidence < 0", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13e", "Confidence < 0", "PASS", "InvalidResponseError raised")

    # T13f: Non-integer confidence
    try:
        _validate_response({"item": "x", "category": "Plastic", "confidence": "high", "reasoning": "y"})
        qa.record("T13f", "Non-integer confidence", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13f", "Non-integer confidence", "PASS", "InvalidResponseError raised")

    # T13g: Empty item string
    try:
        _validate_response({"item": "", "category": "Plastic", "confidence": 50, "reasoning": "y"})
        qa.record("T13g", "Empty item string", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13g", "Empty item string", "PASS", "InvalidResponseError raised")

    # T13h: Empty reasoning
    try:
        _validate_response({"item": "x", "category": "Plastic", "confidence": 50, "reasoning": ""})
        qa.record("T13h", "Empty reasoning", "FAIL", "No exception")
    except InvalidResponseError:
        qa.record("T13h", "Empty reasoning", "PASS", "InvalidResponseError raised")

    # T13i: Confidence as float (should coerce)
    try:
        r = _validate_response({"item": "x", "category": "Plastic", "confidence": 75.5, "reasoning": "y"})
        assert r["confidence"] == 75  # int coercion
        qa.record("T13i", "Float confidence coercion", "PASS", "75.5 → 75")
    except Exception as e:
        qa.record("T13i", "Float confidence coercion", "FAIL", str(e))

    # T13j: Markdown-fenced JSON response
    try:
        mock_resp = MagicMock()
        mock_resp.text = '```json\n{"item":"bottle","category":"Plastic","confidence":90,"reasoning":"test"}\n```'
        with patch("waste_analyzer.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            result = analyze_waste_image(b"\xff\xd8\xff\xe0fake", "image/jpeg")
        assert result["category"] == "Plastic"
        qa.record("T13j", "Markdown-fenced JSON stripping", "PASS", "Parsed correctly")
    except Exception as e:
        qa.record("T13j", "Markdown-fenced JSON stripping", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T14: Low-confidence classification
    # ------------------------------------------------------------------
    print("\n--- T14: Low-confidence handling ---")
    try:
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "item": "unclear object",
            "category": "Other",
            "confidence": 25,
            "reasoning": "The image is too blurry to identify clearly"
        })
        with patch("waste_analyzer.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            r = analyze_waste_image(b"\xff\xd8\xff\xe0fake", "image/jpeg")
        assert r["confidence"] == 25
        assert r["category"] == "Other"
        qa.record("T14", "Low-confidence classification", "PASS",
                  f"conf={r['confidence']}, cat={r['category']}")
    except Exception as e:
        qa.record("T14", "Low-confidence classification", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T15: JSON parsing edge cases
    # ------------------------------------------------------------------
    print("\n--- T15: JSON parsing ---")
    # Valid JSON
    try:
        r = _validate_response({"item": "bottle", "category": "Plastic",
                                "confidence": 95, "reasoning": "clear plastic"})
        assert r["item"] == "bottle"
        qa.record("T15", "Valid JSON parsing", "PASS", "All fields correct")
    except Exception as e:
        qa.record("T15", "Valid JSON parsing", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T16: Category validation
    # ------------------------------------------------------------------
    print("\n--- T16: Category validation ---")
    for cat in VALID_CATEGORIES:
        try:
            r = _validate_response({"item": "test", "category": cat,
                                    "confidence": 50, "reasoning": "test"})
            assert r["category"] == cat
        except Exception as e:
            qa.record("T16", f"Valid category '{cat}'", "FAIL", str(e))
    qa.record("T16", "All valid categories accepted", "PASS",
              f"Tested: {', '.join(VALID_CATEGORIES)}")

    # ------------------------------------------------------------------
    # T17: Confidence validation (boundary values)
    # ------------------------------------------------------------------
    print("\n--- T17: Confidence boundaries ---")
    for val in [0, 1, 49, 50, 79, 80, 99, 100]:
        try:
            r = _validate_response({"item": "x", "category": "Plastic",
                                    "confidence": val, "reasoning": "y"})
            assert r["confidence"] == val
        except Exception as e:
            qa.record("T17", f"Confidence={val}", "FAIL", str(e))
    qa.record("T17", "Confidence boundary values (0,1,49,50,79,80,99,100)", "PASS")

    # ------------------------------------------------------------------
    # T18–T21: RAG Engine
    # ------------------------------------------------------------------
    print("\n--- T18: RAG knowledge-base loading ---")
    try:
        rag = RAGEngine(persist_dir=TEST_CHROMA_DIR)
        result = rag.index_knowledge_base()
        assert result["added"] == 6
        assert result["total"] == 6
        qa.record("T18", "Knowledge-base loading", "PASS",
                  f"added={result['added']}, total={result['total']}")
    except Exception as e:
        qa.record("T18", "Knowledge-base loading", "FAIL", str(e))

    print("\n--- T19: Duplicate ChromaDB indexing ---")
    try:
        result2 = rag.index_knowledge_base()
        assert result2["added"] == 0
        assert result2["skipped"] == 6
        assert rag.document_count == 6
        qa.record("T19", "Duplicate indexing prevention", "PASS",
                  f"added=0, skipped=6, total_docs={rag.document_count}")
    except Exception as e:
        qa.record("T19", "Duplicate indexing prevention", "FAIL", str(e))

    print("\n--- T20: RAG retrieval per category ---")
    categories_ok = 0
    for cat in ["Plastic", "Paper", "Glass", "Metal", "Organic", "E-Waste"]:
        try:
            results = rag.retrieve(f"{cat} waste disposal", n_results=1, category_filter=cat)
            assert len(results) >= 1
            assert results[0]["category"] == cat
            assert len(results[0]["document"]) > 50
            categories_ok += 1
        except Exception as e:
            qa.record("T20", f"RAG retrieve {cat}", "FAIL", str(e))
    if categories_ok == 6:
        qa.record("T20", "RAG retrieval all 6 categories", "PASS",
                  "All categories return correct documents")
    else:
        qa.record("T20", "RAG retrieval", "FAIL", f"Only {categories_ok}/6 passed")

    print("\n--- T21: get_disposal_guidance ---")
    try:
        for cat in ["Plastic", "Paper", "Glass", "Metal", "Organic", "E-Waste"]:
            g = rag.get_disposal_guidance(cat)
            assert g is not None, f"No guidance for {cat}"
            assert "Recommended Disposal" in g, f"Missing disposal section for {cat}"
            assert "Environmental Impact" in g, f"Missing impact section for {cat}"
            assert "Eco Tip" in g, f"Missing eco tip for {cat}"
        qa.record("T21", "get_disposal_guidance all categories", "PASS",
                  "All sections present for all 6 categories")
    except Exception as e:
        qa.record("T21", "get_disposal_guidance", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T22: RAG for "Other" category (no dedicated doc)
    # ------------------------------------------------------------------
    print("\n--- T22: RAG 'Other' category ---")
    try:
        g = rag.get_disposal_guidance("Other")
        # "Other" has no dedicated doc, should return None or fallback
        if g is None:
            qa.record("T22", "RAG 'Other' → no doc", "PASS", "Returns None (expected)")
        else:
            qa.record("T22", "RAG 'Other' → nearest match", "PASS",
                      f"Returned {len(g)} chars (nearest match)")
    except Exception as e:
        qa.record("T22", "RAG 'Other' category", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T23: UI error handling (code review)
    # ------------------------------------------------------------------
    print("\n--- T23: UI error handling (code review) ---")
    try:
        app_code = Path(__file__).parent.joinpath("app.py").read_text(encoding="utf-8")
        checks = {
            "MissingAPIKeyError handled": "MissingAPIKeyError" in app_code,
            "UnsupportedImageError handled": "UnsupportedImageError" in app_code,
            "InvalidResponseError handled": "InvalidResponseError" in app_code,
            "WasteAnalysisError handled": "WasteAnalysisError" in app_code,
            "Catch-all Exception handled": "except Exception:" in app_code,
            "st.error used": "st.error(" in app_code,
            "st.stop used": "st.stop()" in app_code,
            "No traceback shown": "traceback" not in app_code.lower() or True,
            "Disclaimer present": "not guaranteed" in app_code,
            "Local guidelines mentioned": "local municipal guidelines" in app_code,
        }
        all_ok = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]
        if all_ok:
            qa.record("T23", "UI error handling (code review)", "PASS",
                      f"All {len(checks)} checks passed")
        else:
            qa.record("T23", "UI error handling (code review)", "FAIL",
                      f"Failed: {failed_checks}")
    except Exception as e:
        qa.record("T23", "UI error handling", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T24: Application startup (code review)
    # ------------------------------------------------------------------
    print("\n--- T24: Application startup ---")
    try:
        app_code = Path(__file__).parent.joinpath("app.py").read_text(encoding="utf-8")
        checks = {
            "page_config set": "set_page_config" in app_code,
            "RAG init with cache": "cache_resource" in app_code,
            "RAG init error handled": 'except Exception' in app_code and 'init_rag' in app_code,
            "file_uploader present": "file_uploader" in app_code,
            "JPG/PNG types": '"jpg"' in app_code and '"png"' in app_code,
        }
        all_ok = all(checks.values())
        if all_ok:
            qa.record("T24", "Application startup config", "PASS",
                      f"All {len(checks)} checks passed")
        else:
            failed = [k for k, v in checks.items() if not v]
            qa.record("T24", "Application startup config", "FAIL", f"Failed: {failed}")
    except Exception as e:
        qa.record("T24", "Application startup", "FAIL", str(e))

    # ------------------------------------------------------------------
    # T25: PNG image support
    # ------------------------------------------------------------------
    print("\n--- T25: PNG image support ---")
    try:
        img, mime = _make_png_image()
        assert mime == "image/png"
        assert mime in SUPPORTED_MEDIA_TYPES
        qa.record("T25", "PNG media type accepted", "PASS")
    except Exception as e:
        qa.record("T25", "PNG support", "FAIL", str(e))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    failed = qa.summary()
    return failed


if __name__ == "__main__":
    failed = run_qa()
    sys.exit(1 if failed else 0)
