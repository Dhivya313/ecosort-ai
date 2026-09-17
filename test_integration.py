"""
EcoSort AI — End-to-End Integration Test
Verifies the full pipeline: image → Gemini analysis → RAG retrieval.
"""

import io
import sys
import shutil
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFont

from waste_analyzer import analyze_waste_image, VALID_CATEGORIES
from rag_engine import RAGEngine

TEST_CHROMA_DIR = Path(__file__).parent / "chroma_db_test_e2e"


def _create_bottle_image() -> tuple[bytes, str]:
    """Draw a simple plastic-bottle sketch and return (jpeg_bytes, mime)."""
    img = Image.new("RGB", (400, 600), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(130, 180), (270, 500)], radius=30,
                           fill=(135, 206, 235), outline=(70, 130, 180), width=3)
    draw.rounded_rectangle([(170, 100), (230, 190)], radius=8,
                           fill=(135, 206, 235), outline=(70, 130, 180), width=3)
    draw.rounded_rectangle([(165, 70), (235, 110)], radius=6,
                           fill=(30, 144, 255), outline=(20, 100, 200), width=3)
    draw.rounded_rectangle([(140, 280), (260, 400)], radius=10,
                           fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((155, 310), "WATER", fill=(70, 130, 180), font=font)
    draw.text((148, 340), "500 mL", fill=(120, 120, 120), font=font)
    try:
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        small = ImageFont.load_default()
    draw.text((175, 460), "♻ PET 1", fill=(100, 100, 100), font=small)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def main():
    if TEST_CHROMA_DIR.exists():
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    print("=" * 60)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Generate test image
    image_bytes, media_type = _create_bottle_image()
    print(f"\n1. Generated test image ({len(image_bytes):,} bytes, {media_type})")

    # Step 2: Analyze with Gemini
    print("2. Sending to Gemini for analysis…")
    result = analyze_waste_image(image_bytes, media_type)
    print(f"   Item:       {result['item']}")
    print(f"   Category:   {result['category']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   Reasoning:  {result['reasoning']}")

    assert result["category"] in VALID_CATEGORIES
    assert 0 <= result["confidence"] <= 100
    print("   ✓ Analysis validated")

    # Step 3: Initialize RAG and retrieve guidance
    print("\n3. Initializing RAG engine…")
    rag = RAGEngine(persist_dir=TEST_CHROMA_DIR)
    index_result = rag.index_knowledge_base()
    print(f"   Indexed: {index_result}")

    detected_category = result["category"]
    print(f"\n4. Retrieving disposal guidance for '{detected_category}'…")
    guidance = rag.get_disposal_guidance(detected_category)

    assert guidance is not None, "Guidance should not be None"
    assert len(guidance) > 100, "Guidance should be substantial"
    print(f"   ✓ Retrieved {len(guidance)} chars of guidance")

    # Verify key sections exist
    for section in ["Recommended Disposal", "Environmental Impact", "Eco Tip"]:
        found = section in guidance
        status = "✓" if found else "✗"
        print(f"   {status} Section '{section}' present: {found}")

    # Cleanup
    shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("✓ END-TO-END TEST PASSED")
    print("  Image → Gemini analysis → RAG retrieval — all working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
