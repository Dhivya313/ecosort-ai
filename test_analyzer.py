"""
EcoSort AI — Waste Analyzer Test
Generates a synthetic test image with Pillow, sends it to Gemini via
analyze_waste_image(), and validates the structured response.
"""

import io
import os
import sys
import json

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFont

from waste_analyzer import (
    analyze_waste_image,
    MissingAPIKeyError,
    UnsupportedImageError,
    InvalidResponseError,
    WasteAnalysisError,
    VALID_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helper — create a synthetic test image
# ---------------------------------------------------------------------------
def _create_test_image() -> tuple[bytes, str]:
    """Draw a simple 'plastic bottle' sketch and return (jpeg_bytes, mime)."""
    img = Image.new("RGB", (400, 600), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Bottle body (light blue rectangle with rounded feel)
    draw.rounded_rectangle(
        [(130, 180), (270, 500)], radius=30, fill=(135, 206, 235), outline=(70, 130, 180), width=3,
    )
    # Bottle neck
    draw.rounded_rectangle(
        [(170, 100), (230, 190)], radius=8, fill=(135, 206, 235), outline=(70, 130, 180), width=3,
    )
    # Bottle cap
    draw.rounded_rectangle(
        [(165, 70), (235, 110)], radius=6, fill=(30, 144, 255), outline=(20, 100, 200), width=3,
    )
    # Label area
    draw.rounded_rectangle(
        [(140, 280), (260, 400)], radius=10, fill=(255, 255, 255), outline=(200, 200, 200), width=2,
    )
    # Label text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((155, 310), "WATER", fill=(70, 130, 180), font=font)
    draw.text((148, 340), "500 mL", fill=(120, 120, 120), font=font)

    # Recycling symbol text
    try:
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        small_font = ImageFont.load_default()
    draw.text((175, 460), "♻ PET 1", fill=(100, 100, 100), font=small_font)

    # Convert to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_analyze_waste_image():
    """End-to-end test: synthetic bottle image → Gemini → validated JSON."""
    print("=" * 60)
    print("TEST: analyze_waste_image (synthetic plastic bottle image)")
    print("=" * 60)

    image_bytes, media_type = _create_test_image()
    print(f"  ✓ Generated test image: {len(image_bytes):,} bytes, {media_type}")

    print("  → Sending to Gemini API …")
    result = analyze_waste_image(image_bytes, media_type)

    # Pretty-print the result
    print("  ✓ Received response from Gemini:")
    print(json.dumps(result, indent=4))

    # Validate structure
    assert isinstance(result, dict), "Result should be a dict"
    for key in ("item", "category", "confidence", "reasoning"):
        assert key in result, f"Missing key: {key}"

    assert result["category"] in VALID_CATEGORIES, (
        f"Invalid category: {result['category']}"
    )
    assert isinstance(result["confidence"], int), "Confidence should be int"
    assert 0 <= result["confidence"] <= 100, (
        f"Confidence out of range: {result['confidence']}"
    )
    assert isinstance(result["item"], str) and result["item"].strip()
    assert isinstance(result["reasoning"], str) and result["reasoning"].strip()

    print("  ✓ All assertions passed")
    print()
    return result


def test_unsupported_media_type():
    """Ensure UnsupportedImageError is raised for bad media types."""
    print("=" * 60)
    print("TEST: unsupported media type")
    print("=" * 60)

    try:
        analyze_waste_image(b"fake-image-data", "image/bmp")
        print("  ✗ FAIL — no exception raised")
        return False
    except UnsupportedImageError as exc:
        print(f"  ✓ Correctly raised UnsupportedImageError: {exc}")
        return True


def test_empty_image():
    """Ensure UnsupportedImageError is raised for empty image data."""
    print("=" * 60)
    print("TEST: empty image data")
    print("=" * 60)

    try:
        analyze_waste_image(b"", "image/jpeg")
        print("  ✗ FAIL — no exception raised")
        return False
    except UnsupportedImageError as exc:
        print(f"  ✓ Correctly raised UnsupportedImageError: {exc}")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    passed = 0
    failed = 0

    # Test 1: unsupported media type (no API call needed)
    if test_unsupported_media_type():
        passed += 1
    else:
        failed += 1

    # Test 2: empty image data (no API call needed)
    if test_empty_image():
        passed += 1
    else:
        failed += 1

    # Test 3: full end-to-end with Gemini
    try:
        test_analyze_waste_image()
        passed += 1
    except MissingAPIKeyError as exc:
        print(f"  ✗ SKIPPED (no API key): {exc}")
    except WasteAnalysisError as exc:
        print(f"  ✗ FAIL: {exc}")
        failed += 1
    except Exception as exc:
        print(f"  ✗ FAIL (unexpected): {type(exc).__name__}: {exc}")
        failed += 1

    # Summary
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed else 0)
