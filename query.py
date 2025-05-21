#!/usr/bin/env python
#!pip install -U langchain-chroma
#!pip install -U langchain-ollama
import argparse
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
PROMPT = ChatPromptTemplate.from_template("""
Answer the question **only** with information in the context.

{context}

---
Question: {question}
""")

def ask(question: str):
    db   = Chroma(persist_directory=CHROMA_PATH,
                  embedding_function=get_embedding_function())
    docs = db.similarity_search_with_score(question, k=5)

    context = "\n\n---\n\n".join(d.page_content for d, _ in docs)
    prompt  = PROMPT.format(context=context, question=question)

    llm     = OllamaLLM(model="gemma:latest")
    answer  = llm.invoke(prompt)

    print("\n✅  Answer\n", answer, "\n")
    print("🔗 Sources:", [d.metadata["id"] for d, _ in docs])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    args = p.parse_args()
    ask(" ".join(args.question))
