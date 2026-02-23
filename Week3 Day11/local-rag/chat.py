from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import DB_PATH, EMBEDDING_MODEL, LLM_MODEL, TOP_K, SYSTEM_PROMPT

def main():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    llm = OllamaLLM(model=LLM_MODEL)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT + "\nQuestion:\n{question}\n\nAnswer:"
    )

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        docs = db.similarity_search(question, k=TOP_K)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(final_prompt)

        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()