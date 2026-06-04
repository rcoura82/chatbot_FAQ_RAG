from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = BASE_DIR / "data"
DEFAULT_HISTORY_PATH = BASE_DIR / "chat_history.jsonl"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
# Mantém as respostas curtas e objetivas para um FAQ de terminal.
MAX_NEW_TOKENS = 128
# Inclui apenas contextos com pelo menos 80% da pontuação do melhor resultado.
SIMILARITY_THRESHOLD_RATIO = 0.8
MIN_VALID_SCORE = 0.0
STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
}


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    content: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    content: str
    score: float


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def load_corpus(corpus_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        for paragraph in (item.strip() for item in text.split("\n\n")):
            if paragraph:
                chunks.append(DocumentChunk(source=path.name, content=paragraph))
    if not chunks:
        raise ValueError(f"Nenhum arquivo .txt encontrado em {corpus_dir}")
    return chunks


def build_prompt(question: str, contexts: Sequence[RetrievedChunk]) -> str:
    formatted_context = "\n".join(
        f"[{index}] {chunk.content}" for index, chunk in enumerate(contexts, start=1)
    )
    return (
        "Você é um assistente especializado no Guia de Festas Juninas do RJ.\n"
        "Responda em português usando apenas o contexto fornecido.\n"
        "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação no guia.\n\n"
        f"Contexto:\n{formatted_context}\n\n"
        f"Pergunta: {question}\n"
        "Resposta:"
    )


class KeywordRetriever:
    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        term_frequencies = [Counter(tokenize(chunk.content)) for chunk in self._chunks]
        self._inverse_document_frequency = self._build_inverse_document_frequency(term_frequencies)
        self._vectors = [self._weight_vector(vector) for vector in term_frequencies]
        self._norms = [self._norm(vector) for vector in self._vectors]

    @staticmethod
    def _build_inverse_document_frequency(
        term_frequencies: Sequence[Counter[str]],
    ) -> dict[str, float]:
        total_documents = len(term_frequencies)
        document_frequencies: Counter[str] = Counter()
        for vector in term_frequencies:
            document_frequencies.update(vector.keys())
        return {
            term: math.log((1 + total_documents) / (1 + count)) + 1
            for term, count in document_frequencies.items()
        }

    def _weight_vector(self, vector: Counter[str]) -> dict[str, float]:
        return {
            term: frequency * self._inverse_document_frequency.get(term, 1.0)
            for term, frequency in vector.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine_similarity(
        left_vector: dict[str, float],
        left_norm: float,
        right_vector: dict[str, float],
        right_norm: float,
    ) -> float:
        if not left_norm or not right_norm:
            return 0.0
        shared_terms = set(left_vector) & set(right_vector)
        dot_product = sum(left_vector[term] * right_vector[term] for term in shared_terms)
        return dot_product / (left_norm * right_norm)

    def search(self, question: str, top_k: int = 1) -> list[RetrievedChunk]:
        question_vector = self._weight_vector(Counter(tokenize(question)))
        question_norm = self._norm(question_vector)
        ranked_chunks = sorted(
            (
                RetrievedChunk(
                    source=chunk.source,
                    content=chunk.content,
                    score=self._cosine_similarity(question_vector, question_norm, vector, norm),
                )
                for chunk, vector, norm in zip(self._chunks, self._vectors, self._norms)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return [chunk for chunk in ranked_chunks[:top_k] if chunk.score > 0]


class FaissRetriever:
    def __init__(self, chunks: Sequence[DocumentChunk], model_name: str) -> None:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._faiss = faiss
        self._np = np
        self._chunks = list(chunks)
        self._embedder = SentenceTransformer(model_name)
        embeddings = self._embedder.encode(
            [chunk.content for chunk in self._chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, question: str, top_k: int = 1) -> list[RetrievedChunk]:
        question_embedding = self._embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self._index.search(question_embedding, top_k)
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self._chunks[index]
            results.append(
                RetrievedChunk(source=chunk.source, content=chunk.content, score=float(score))
            )
        return results


class TemplateGenerator:
    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "Não encontrei essa informação no guia de festas juninas do RJ."
        best_context = contexts[0].content
        return (
            f"Encontrei a seguinte informação no guia: {best_context}\n\n"
            f"Pergunta recebida: {question}"
        )


class HuggingFaceGenerator:
    def __init__(self, model_name: str) -> None:
        from transformers import pipeline  # type: ignore

        self._pipeline = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        if not contexts:
            return "Não encontrei essa informação no guia de festas juninas do RJ."
        prompt = build_prompt(question, contexts)
        result = self._pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return result[0]["generated_text"].strip()


class FAQRAGChatbot:
    def __init__(
        self,
        corpus_dir: Path = DEFAULT_CORPUS_DIR,
        history_path: Path = DEFAULT_HISTORY_PATH,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model_name: str = "google/flan-t5-base",
    ) -> None:
        self._corpus_dir = corpus_dir
        self._history_path = history_path
        self._chunks = load_corpus(corpus_dir)
        self._retriever, retriever_mode = self._build_retriever(embedding_model_name)
        self._generator, final_runtime_mode = self._build_generator(
            generation_model_name, retriever_mode
        )
        self.runtime_mode = final_runtime_mode

    def _build_retriever(
        self, embedding_model_name: str
    ) -> tuple[KeywordRetriever | FaissRetriever, str]:
        try:
            retriever = FaissRetriever(self._chunks, embedding_model_name)
            return retriever, "huggingface+faiss"
        except (ImportError, OSError, RuntimeError, ValueError):
            return KeywordRetriever(self._chunks), "fallback"

    def _build_generator(
        self, generation_model_name: str, runtime_mode: str
    ) -> tuple[TemplateGenerator | HuggingFaceGenerator, str]:
        if runtime_mode == "huggingface+faiss":
            try:
                return HuggingFaceGenerator(generation_model_name), runtime_mode
            except (ImportError, OSError, RuntimeError, ValueError):
                return TemplateGenerator(), "fallback"
        return TemplateGenerator(), runtime_mode

    def answer_question(self, question: str, top_k: int = 1) -> str:
        contexts = self._filter_contexts(self._retriever.search(question, top_k=top_k))
        answer = self._generator.generate(question, contexts)
        if contexts:
            sources = ", ".join(dict.fromkeys(chunk.source for chunk in contexts))
            answer = f"{answer}\n\nFontes consultadas: {sources}"
        self._append_history(question, answer)
        return answer

    @staticmethod
    def _filter_contexts(contexts: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not contexts:
            return []
        if len(contexts) == 1 or contexts[0].score <= MIN_VALID_SCORE:
            return list(contexts)
        threshold = contexts[0].score * SIMILARITY_THRESHOLD_RATIO
        filtered_contexts = [chunk for chunk in contexts if chunk.score >= threshold]
        return filtered_contexts or [contexts[0]]

    def _append_history(self, question: str, answer: str) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(
                json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n"
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chatbot de FAQ com RAG simples.")
    parser.add_argument("--question", help="Pergunta única para o chatbot.")
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=1,
        help="Quantidade de trechos recuperados.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Diretório com arquivos .txt do corpus.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Arquivo JSONL para salvar o histórico.",
    )
    return parser


def interactive_chat(chatbot: FAQRAGChatbot, top_k: int) -> None:
    print(f"Modo de execução: {chatbot.runtime_mode}")
    print("Digite sua pergunta sobre o guia de festas juninas do RJ ou 'sair' para encerrar.")
    while True:
        question = input("Você: ").strip()
        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Até logo!")
            return
        print(f"Chatbot: {chatbot.answer_question(question, top_k=top_k)}\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chatbot = FAQRAGChatbot(corpus_dir=args.corpus_dir, history_path=args.history_file)
    if args.question:
        print(chatbot.answer_question(args.question, top_k=args.top_k))
        return 0
    interactive_chat(chatbot, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
