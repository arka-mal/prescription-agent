"""
run_batch.py
------------
Batch processor for prescription images.
Runs OCR + Layout agents on all images in the dataset folder
and writes raw results to batch-output/results.csv

Usage:
    python run_batch.py

Requirements:
    - .env file in root folder with GROQ_API_KEY and GCP_CREDENTIALS_PATH
    - dataset/ folder with images named 1.jpg, 2.jpg ... 129.jpg
    - venv activated
"""

import os
import sys
import csv
import time
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

if not GCP_CREDENTIALS_PATH or not os.path.exists(GCP_CREDENTIALS_PATH):
    print(f"ERROR: GCP_CREDENTIALS_PATH not found or file does not exist: {GCP_CREDENTIALS_PATH}")
    sys.exit(1)

# ── Load GCP credentials from JSON file ───────────────────────────────────────
with open(GCP_CREDENTIALS_PATH, "r") as f:
    GCP_CREDENTIALS_INFO = json.load(f)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR  = ROOT_DIR / "batch-output"
OUTPUT_CSV  = OUTPUT_DIR / "results.csv"
LOG_FILE    = OUTPUT_DIR / "run_log.txt"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Import agents ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT_DIR))
from agents.ocr_agent import run_ocr_agent
from agents.layout_agent import run_layout_agent

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"
DELAY_SECONDS = 2        # delay between images to avoid rate limiting
TOTAL_IMAGES  = 129

# All 7 zone types
ZONES = [
    "prescriber_footer",
    "patient_header",
    "drug_list",
    "dosage_column",
    "duration_column",
    "clinical_notes",
    "annotation",
]

# ── CSV Header ────────────────────────────────────────────────────────────────
CSV_HEADER = [
    "image_name",
    "ocr_blocks",
    "ocr_confidence",
    "word_count",
    "mixed_script",
    "scripts_detected",
    "handwritten_blocks",
    "segments_found",
    "layout_confidence",
    # zone identified (1/0)
    "zone_prescriber",
    "zone_patient",
    "zone_drug",
    "zone_dosage",
    "zone_duration",
    "zone_clinical",
    "zone_annotation",
    # bbox assigned (1/0)
    "bbox_prescriber",
    "bbox_patient",
    "bbox_drug",
    "bbox_dosage",
    "bbox_duration",
    "bbox_clinical",
    "bbox_annotation",
    # per-segment confidence (high/medium/low or empty)
    "conf_prescriber",
    "conf_patient",
    "conf_drug",
    "conf_dosage",
    "conf_duration",
    "conf_clinical",
    "conf_annotation",
    "status",       # success / ocr_failed / layout_failed
    "error_msg",    # empty if success
]


def get_image_files():
    """Get all image files from dataset folder, sorted numerically."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
    files = []
    for i in range(1, TOTAL_IMAGES + 1):
        for ext in extensions:
            candidate = DATASET_DIR / f"{i}{ext}"
            if candidate.exists():
                files.append((i, candidate))
                break
    return files


def process_image(image_path: Path):
    """
    Run OCR + Layout on one image.
    Returns a dict of all metrics for this image.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # ── Stage 1: OCR ─────────────────────────────────────────────────────────
    ocr = run_ocr_agent(
        image_bytes,
        credentials_info=GCP_CREDENTIALS_INFO,
    )

    # Count handwritten blocks
    handwritten_count = sum(
        1 for b in ocr.blocks
        if (b.get("is_handwritten_guess") if isinstance(b, dict) else getattr(b, "is_handwritten_guess", False))
    )

    # ── Stage 2: Layout ───────────────────────────────────────────────────────
    layout = run_layout_agent(
        raw_ocr_text=ocr.raw_text,
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        ocr_blocks=ocr.blocks,
    )

    # ── Extract per-zone metrics ──────────────────────────────────────────────
    # Build lookup: label -> segment object (take first match if duplicates)
    seg_lookup = {}
    for seg in layout.segments:
        if seg.label not in seg_lookup:
            seg_lookup[seg.label] = seg

    zone_identified = {}
    zone_bbox       = {}
    zone_conf       = {}

    for zone_key, zone_label in zip(
        ["prescriber_footer", "patient_header", "drug_list",
         "dosage_column", "duration_column", "clinical_notes", "annotation"],
        ZONES
    ):
        seg = seg_lookup.get(zone_label)
        zone_identified[zone_label] = 1 if seg else 0
        zone_bbox[zone_label]       = 1 if (seg and seg.bounding_box) else 0
        zone_conf[zone_label]       = seg.confidence.value if seg else ""

    return {
        "ocr_blocks":        len(ocr.blocks),
        "ocr_confidence":    ocr.overall_confidence.value,
        "word_count":        ocr.word_count,
        "mixed_script":      1 if len(ocr.mixed_script_flags) > 0 else 0,
        "scripts_detected":  "|".join(ocr.detected_scripts),
        "handwritten_blocks": handwritten_count,
        "segments_found":    len(layout.segments),
        "layout_confidence": layout.overall_confidence.value,
        "zone_identified":   zone_identified,
        "zone_bbox":         zone_bbox,
        "zone_conf":         zone_conf,
    }


def write_row(writer, image_name, data, status="success", error_msg=""):
    """Write one row to the CSV."""
    if status == "success":
        zi = data["zone_identified"]
        zb = data["zone_bbox"]
        zc = data["zone_conf"]
        row = [
            image_name,
            data["ocr_blocks"],
            data["ocr_confidence"],
            data["word_count"],
            data["mixed_script"],
            data["scripts_detected"],
            data["handwritten_blocks"],
            data["segments_found"],
            data["layout_confidence"],
            zi.get("prescriber_footer", 0),
            zi.get("patient_header", 0),
            zi.get("drug_list", 0),
            zi.get("dosage_column", 0),
            zi.get("duration_column", 0),
            zi.get("clinical_notes", 0),
            zi.get("annotation", 0),
            zb.get("prescriber_footer", 0),
            zb.get("patient_header", 0),
            zb.get("drug_list", 0),
            zb.get("dosage_column", 0),
            zb.get("duration_column", 0),
            zb.get("clinical_notes", 0),
            zb.get("annotation", 0),
            zc.get("prescriber_footer", ""),
            zc.get("patient_header", ""),
            zc.get("drug_list", ""),
            zc.get("dosage_column", ""),
            zc.get("duration_column", ""),
            zc.get("clinical_notes", ""),
            zc.get("annotation", ""),
            status,
            error_msg,
        ]
    else:
        # Failed row — fill with blanks
        row = [image_name] + [""] * (len(CSV_HEADER) - 3) + [status, error_msg]

    writer.writerow(row)


def main():
    image_files = get_image_files()

    if not image_files:
        print(f"ERROR: No images found in {DATASET_DIR}")
        print("Expected files named 1.jpg, 2.jpg ... 129.jpg")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Prescription Batch Processor")
    print(f"{'='*60}")
    print(f"  Dataset folder : {DATASET_DIR}")
    print(f"  Images found   : {len(image_files)}")
    print(f"  Output CSV     : {OUTPUT_CSV}")
    print(f"  Delay          : {DELAY_SECONDS}s between images")
    print(f"  Model          : {GROQ_MODEL}")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count    = 0
    failed_images = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile, \
         open(LOG_FILE,   "w", encoding="utf-8") as logfile:

        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)

        for idx, (img_num, img_path) in enumerate(image_files, start=1):
            image_name = img_path.name
            prefix = f"[{idx:03d}/{len(image_files)}] {image_name}"

            print(f"{prefix} → processing...", end="", flush=True)

            try:
                data = process_image(img_path)

                # Build console summary
                ocr_conf  = data["ocr_confidence"].upper()
                lay_conf  = data["layout_confidence"].upper()
                blocks    = data["ocr_blocks"]
                segs      = data["segments_found"]
                scripts   = data["scripts_detected"]
                mixed     = "MIXED" if data["mixed_script"] else ""

                bbox_yes  = sum(data["zone_bbox"].values())
                bbox_tot  = sum(data["zone_identified"].values())

                status_line = (
                    f" OCR:{blocks}blk/{ocr_conf} | "
                    f"Layout:{segs}seg/{lay_conf} | "
                    f"BBox:{bbox_yes}/{bbox_tot} | "
                    f"Scripts:{scripts} {mixed}"
                )
                print(f"\r{prefix} ✓{status_line}")

                log_entry = f"[{idx:03d}] {image_name}: SUCCESS | {status_line}\n"
                logfile.write(log_entry)

                write_row(writer, image_name, data, status="success")
                csvfile.flush()  # write immediately in case of crash

                success_count += 1

            except Exception as e:
                error_msg = str(e)
                short_err = error_msg[:80].replace("\n", " ")
                print(f"\r{prefix} ✗ FAILED: {short_err}")

                log_entry = f"[{idx:03d}] {image_name}: FAILED | {error_msg}\n{traceback.format_exc()}\n"
                logfile.write(log_entry)

                write_row(writer, image_name, {}, status="failed", error_msg=short_err)
                csvfile.flush()

                fail_count    += 1
                failed_images.append(image_name)

            # Delay between images (skip after last one)
            if idx < len(image_files):
                time.sleep(DELAY_SECONDS)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed : {len(image_files)}")
    print(f"  Success   : {success_count}")
    print(f"  Failed    : {fail_count}")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"  Log       : {LOG_FILE}")

    if failed_images:
        print(f"\n  Failed images:")
        for name in failed_images:
            print(f"    - {name}")

    print(f"{'='*60}\n")
    print("Run compute_metrics.py next to generate charts and summary.")


if __name__ == "__main__":
    main()
