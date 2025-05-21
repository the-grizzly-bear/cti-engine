#!/usr/bin/env python
import argparse, os, shutil, sys
import contextlib
#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from helpers import clean_str
#from list import list_pdfs_in_db

DATA_PATH   = "data" 
CHROMA_PATH = "chroma"

def clear_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# def load_docs():
#     return PyPDFDirectoryLoader(DATA_PATH).load()
def load_docs():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    try:
        with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
            return loader.load()
    except Exception as e:
        print(f"[load_docs] Failed to load PDF(s): {e}")
        return []

def split_docs(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len
    )
    return splitter.split_documents(docs)


def tag_chunks(chunks: list[Document]):
    last_page, idx = None, 0
    for c in chunks:
        pid = f"{c.metadata['source']}:{c.metadata['page']}"
        idx = idx + 1 if pid == last_page else 0
        c.metadata["id"] = f"{pid}:{idx}"
        last_page = pid
    return chunks


from helpers import clean_str

def ingest(reset: bool = False) -> None:
    if reset:
        print("✨ Resetting DB")
        clear_db()

    # ── load / split / tag ───────────────────────────────────────────
    chunks = tag_chunks(split_docs(load_docs()))

    # ── open (or create) Chroma DB ──────────────────────────────────
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    existing_ids = set(db.get(include=[])["ids"])
    new_chunks   = []

    for c in chunks:
        if c.metadata["id"] not in existing_ids:
            # clean text & metadata so UTF‑8 encode never fails
            c.page_content = clean_str(c.page_content)
            c.metadata     = {k: clean_str(str(v)) for k, v in c.metadata.items()}
            new_chunks.append(c)

    # ── add & persist ───────────────────────────────────────────────
    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new chunks")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        db.persist()
    else:
        print("✅ Nothing new to add")

def list_pdfs_in_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    all_docs = db.get()
    sources = set()
    for meta in all_docs.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    print("PDFs in database:")
    for src in sorted(sources):
        print("-", src)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--reset", action="store_true")
#     ap.add_argument("--list", action="store_true", help="List PDFs in the database")
#     args = ap.parse_args()
#     if args.list:
#         list_pdfs_in_db()
#     else:
#         ingest(reset=args.reset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--list", action="store_true", help="List PDFs in the database")
    args = ap.parse_args()
    
    if args.list:
        list_pdfs_in_db()
    else:
        ingest(reset=args.reset)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--reset", action="store_true")
#     ingest(**vars(ap.parse_args()))
