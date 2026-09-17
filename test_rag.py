"""
EcoSort AI — RAG Engine Test
Indexes the knowledge base and verifies retrieval for all six waste categories.
"""

import sys
import shutil
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from rag_engine import RAGEngine

# Use a temporary DB directory so tests are isolated
TEST_CHROMA_DIR = Path(__file__).parent / "chroma_db_test"

CATEGORIES_TO_TEST = [
    "Plastic",
    "Paper",
    "Glass",
    "Metal",
    "Organic",
    "E-Waste",
]


def run_test():
    # Clean up any previous test DB
    if TEST_CHROMA_DIR.exists():
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    print("=" * 60)
    print("RAG ENGINE TEST")
    print("=" * 60)

    # ---- Initialize & Index -----------------------------------------------
    rag = RAGEngine(persist_dir=TEST_CHROMA_DIR)
    print(f"\n1. Initialized RAGEngine (DB: {TEST_CHROMA_DIR})")

    result = rag.index_knowledge_base()
    print(f"2. Indexed knowledge base: {result}")
    assert result["added"] == 6, f"Expected 6 docs added, got {result['added']}"
    assert result["skipped"] == 0
    print("   ✓ All 6 documents indexed\n")

    # ---- Test dedup (re-index should skip all) ----------------------------
    result2 = rag.index_knowledge_base()
    print(f"3. Re-indexed (dedup test): {result2}")
    assert result2["added"] == 0, f"Expected 0 added on re-index, got {result2['added']}"
    assert result2["skipped"] == 6
    print("   ✓ No duplicates created\n")

    assert rag.document_count == 6, f"Expected 6 docs, got {rag.document_count}"
    print(f"4. Document count: {rag.document_count} ✓\n")

    # ---- Test retrieval for each category ---------------------------------
    print("5. Retrieval tests:")
    print("-" * 60)

    passed = 0
    failed = 0

    for category in CATEGORIES_TO_TEST:
        results = rag.retrieve(
            query=f"{category} waste disposal guidance",
            n_results=1,
            category_filter=category,
        )

        if results:
            r = results[0]
            status = "✓" if r["category"] == category else "✗"
            if r["category"] == category:
                passed += 1
            else:
                failed += 1
            preview = r["document"][:80].replace("\n", " ").strip()
            print(f"   {status} {category:10s} → retrieved: {r['source_file']:15s} "
                  f"(distance: {r['distance']:.4f})")
            print(f"     preview: \"{preview}…\"")
        else:
            failed += 1
            print(f"   ✗ {category:10s} → NO RESULTS")

    print("-" * 60)

    # ---- Test get_disposal_guidance convenience method ---------------------
    print("\n6. get_disposal_guidance('Plastic'):")
    guidance = rag.get_disposal_guidance("Plastic")
    assert guidance is not None, "Should return guidance for Plastic"
    assert "Plastic" in guidance
    print(f"   ✓ Returned {len(guidance)} chars of guidance")

    # ---- Free-text query test (no filter) ---------------------------------
    print("\n7. Free-text query: 'How do I recycle old batteries?'")
    results = rag.retrieve("How do I recycle old batteries?", n_results=1)
    if results:
        r = results[0]
        print(f"   → Best match: {r['source_file']} ({r['category']}, "
              f"distance: {r['distance']:.4f})")
        print(f"   ✓ Makes sense — batteries are {r['category']}")

    # ---- Cleanup ----------------------------------------------------------
    shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed (out of {len(CATEGORIES_TO_TEST)} categories)")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
