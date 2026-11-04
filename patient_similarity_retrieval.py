"""CLI: python patient_similarity_retrieval.py PATH_TO_INPUT_DATA_FOLDER"""
import argparse
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Retrieve similar patients using TF-IDF and BM25")
    parser.add_argument("input_dir", help="Folder containing patient JSON files")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-features", type=int, default=30000)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.max_features)


if __name__ == "__main__":
    main()
