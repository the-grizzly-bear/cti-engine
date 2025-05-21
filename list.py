def list_pdfs_in_db():
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    # Get all document metadata
    all_docs = db.get()
    sources = set()
    for meta in all_docs.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    print("PDFs in database:")
    for src in sorted(sources):
        print("-", src)

# To use from command line, add:
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--list", action="store_true", help="List PDFs in the database")
    args = ap.parse_args()
    if args.list:
        list_pdfs_in_db()
    else:
        ingest(reset=args.reset)