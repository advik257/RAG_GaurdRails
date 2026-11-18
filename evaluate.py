import argparse
from datasets import Dataset

TEST_DATA = [
    {
        "question": "What is RAG?",
        "ground_truth": "Retrieval-Augmented Generation combines a retriever with a generator LLM to answer using documents.",
    },
    {
        "question": "What are guardrails in RAG?",
        "ground_truth": "Guardrails are safety checks on input and output to block injection, PII, toxic content.",
    },
    {
        "question": "What is ChromaDB used for?",
        "ground_truth": "ChromaDB is a vector database to store embeddings and retrieve similar chunks.",
    },
]

def run_ragas():
    # IMPORTANT: Tell RAGAS to use Groq + Cohere, not OpenAI
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_cohere import CohereEmbeddings
    from config import Config
    from rag import RAGSystem

    # Groq for evaluation
    evaluator_llm = LangchainLLMWrapper(ChatGroq(
        model=Config.LLM_MODEL,
        temperature=0,
        groq_api_key=Config.GROQ_API_KEY
    ))
    evaluator_emb = LangchainEmbeddingsWrapper(CohereEmbeddings(
        model=Config.EMBEDDING_MODEL,
        cohere_api_key=Config.COHERE_API_KEY
    ))

    rag = RAGSystem()
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATA:
        print(f"Evaluating: {item['question']}")
        res = rag.ask(item["question"])
        docs = rag.retriever.invoke(item["question"])
        ctx = [d.page_content for d in docs]

        questions.append(item["question"])
        answers.append(res["answer"])
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )
    print("\n=== RAGAS RESULTS ===")
    print(result)
    print(result.to_pandas())

if __name__ == "__main__":
    run_ragas()