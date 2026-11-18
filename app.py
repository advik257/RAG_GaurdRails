from rag import RAGSystem

if __name__ == "__main__":
    rag = RAGSystem()
    print("RAG with Guardrails Ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            break
        res = rag.ask(q)
        # Show what guardrail did
        if res.get("cleaned_query") != q:
            print(f"[Guardrail] Cleaned Query Sent to LLM: {res.get('cleaned_query')}")
        print(f"\nAI: {res['answer']}")
        if res['sources']:
            print(f"Sources: {res['sources']}")