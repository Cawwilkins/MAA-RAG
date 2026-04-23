from __future__ import annotations
import json
import traceback
import time
from pathlib import Path
from typing import Any
from llama_index.core import VectorStoreIndex
from setup_index.feed_documents_upsert import feed_documents
from setup_index.doc_embed_store import doc_embed_store
from setup_index.file_utils import get_source_id
from setup_index.update_metadata_facets import (
    initialize_empty_metadata_files,
    update_metadata_facets_for_files,
)
from config import DOCS_DIR, STORED_FILES_PATH, DEFAULT_MAX_UPSERT_FILES, ACCEPTED_FILE_TYPES, METADATA_FACETS_PATH, METADATA_SOURCE_CONTRIBUTIONS_PATH



# Gets the state of each file, such as path, source id, size, and last edit time
def get_file_state(file_path: str | Path, root_path: str | Path) -> dict[str, Any]:
    path = Path(file_path).resolve()
    stat = path.stat()
    return {
        "source_id": get_source_id(path, root_path),
        "file_path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


# Brings the json of stored files into RAM
def load_stored_files(json_path: Path) -> dict[str, dict[str, Any]]:
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        print(f"Warning: {json_path} did not contain a dictionary. Resetting.")
        return {}

    except Exception as e:
        print(f"Warning: failed to load {json_path}: {e}")
        return {}


# Write all the files that will be stored to json file
    # This is a bit easy to break as if embedding fails, json file still contains the names
def save_stored_files(json_path: Path, data: dict[str, dict[str, Any]]) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Check the state of each file against the json list
def file_state_changed(old_state: dict[str, Any], new_state: dict[str, Any]) -> bool:
    return (
        old_state.get("size") != new_state.get("size")
        or old_state.get("mtime_ns") != new_state.get("mtime_ns")
        or old_state.get("file_path") != new_state.get("file_path")
    )


# Parses docs in given path, checks if they need to be added to store, if so adds to list of files to be inserted
def parse_documents(
    root_path: str | Path,
    json_path: Path = STORED_FILES_PATH,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    # Took .02 seconds with 41 docs, meaning took that long to compile list of docs to be added
    parse_start = time.perf_counter()

    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    stored_files = load_stored_files(json_path)
    files_to_upsert: list[str] = []
    count = 0

    user_input = input(
        "> MAA Assistant: How many files would you like to upsert "
        "(40 will take ~35 minutes)? Press Enter for default: "
    )

    if user_input.strip() == "":
        max_upsert_files = DEFAULT_MAX_UPSERT_FILES
    else:
        try:
            max_upsert_files = int(user_input)
        except ValueError:
            print("Invalid input. Using default of 40.")
            max_upsert_files = DEFAULT_MAX_UPSERT_FILES
            
    for path in root.rglob("*"):
        if count >= max_upsert_files:
            print("> MAA Assistant: Stopping embedding early because max file count reached.")
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in ACCEPTED_FILE_TYPES:
            continue
        try:
            curr_doc = get_file_state(path, root)

            source_id = curr_doc["source_id"]

            if source_id not in stored_files:
                stored_files[source_id] = {
                    "file_path": curr_doc["file_path"],
                    "size": curr_doc["size"],
                    "mtime_ns": curr_doc["mtime_ns"],
                }
                files_to_upsert.append(curr_doc["file_path"])
                count += 1
            else:
                old_state = stored_files[source_id]

                if file_state_changed(old_state, curr_doc):
                    stored_files[source_id] = {
                        "file_path": curr_doc["file_path"],
                        "size": curr_doc["size"],
                        "mtime_ns": curr_doc["mtime_ns"],
                    }
                    files_to_upsert.append(curr_doc["file_path"])
                    count += 1

        except Exception as e:
            print(f"Error while processing file {path}: {e}")
            traceback.print_exc()

    print(f"Number of files added is {count}")
    save_stored_files(json_path, stored_files)

    parse_elapsed = time.perf_counter() - parse_start
    print(f"parse_documents took {parse_elapsed:.2f} seconds")

    return files_to_upsert, stored_files


# Runs the parser, feeds docs, and upserts to doc store
def create_index(filepath: str | Path) -> VectorStoreIndex | None:
    total_start = time.perf_counter()

    try:
        print("About to parse through documents...")
        initialize_empty_metadata_files(
            METADATA_FACETS_PATH,
            METADATA_SOURCE_CONTRIBUTIONS_PATH,
        )

        files_to_upsert, stored_files = parse_documents(filepath)

        if not files_to_upsert:
            total_elapsed = time.perf_counter() - total_start
            print("No new or modified files found.")
            print(f"Total runtime: {total_elapsed:.2f} seconds")
            return None

        print("Loading documents...")
        for files in files_to_upsert:
            print(files)

        load_start = time.perf_counter()
        print("About to feed documents")
        documents, rich_metadata_map = feed_documents(files_to_upsert, docs_root=filepath)
        load_elapsed = time.perf_counter() - load_start
        print(f"feed_documents took {load_elapsed:.2f} seconds")

        try:
            print("Embedding and storing documents...")
            embed_start = time.perf_counter()
            index = doc_embed_store(documents, rich_metadata_map)
            update_metadata_facets_for_files(
                rich_metadata_map=rich_metadata_map,
                files_to_upsert=files_to_upsert,
                docs_root=filepath,
                stored_files=stored_files,
                stored_files_path=STORED_FILES_PATH,
                facets_path=METADATA_FACETS_PATH,
                source_contributions_path=METADATA_SOURCE_CONTRIBUTIONS_PATH,
            )
            embed_elapsed = time.perf_counter() - embed_start
            print(f"doc_embed_store took {embed_elapsed:.2f} seconds")

            total_elapsed = time.perf_counter() - total_start
            print("Documents embedded and stored successfully.")
            print(f"Total runtime: {total_elapsed:.2f} seconds")

            return index

        except Exception as e:
            print(f"Error occurred while embedding documents: {e}")
            traceback.print_exc()
            total_elapsed = time.perf_counter() - total_start
            print(f"Total runtime before failure: {total_elapsed:.2f} seconds")
            return None

    except Exception as e:
        print(f"Error occurred while loading documents: {e}")
        traceback.print_exc()
        total_elapsed = time.perf_counter() - total_start
        print(f"Total runtime before failure: {total_elapsed:.2f} seconds")
        return None




if __name__ == "__main__":
    #total pipeline is 35 mins
    program_start = time.perf_counter()
    index = create_index(DOCS_DIR)
    program_elapsed = time.perf_counter() - program_start

    if index is not None:
        print("Index created successfully.")
    else:
        print("No index created.")

    print(f"Program finished in {program_elapsed:.2f} seconds")