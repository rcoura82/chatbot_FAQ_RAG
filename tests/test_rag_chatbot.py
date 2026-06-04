import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import rag_chatbot
from rag_chatbot import (
    FAQRAGChatbot,
    KeywordRetriever,
    build_prompt,
    get_base_dir,
    get_default_history_path,
    load_corpus,
)


class RagChatbotTests(unittest.TestCase):
    def test_get_default_history_path_uses_custom_home(self) -> None:
        with patch.dict("os.environ", {"CHATBOT_FAQ_RAG_HOME": "/tmp/custom-home"}):
            history_path = get_default_history_path()

        self.assertEqual(Path("/tmp/custom-home/chat_history.jsonl"), history_path)

    def test_get_default_history_path_uses_user_home_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("rag_chatbot.Path.home", return_value=Path("/tmp/user-home")):
                history_path = get_default_history_path()

        self.assertEqual(Path("/tmp/user-home/.chatbot_faq_rag/chat_history.jsonl"), history_path)

    def test_get_base_dir_prefers_meipass_when_frozen(self) -> None:
        with patch.object(rag_chatbot.sys, "frozen", True, create=True):
            with patch.object(rag_chatbot.sys, "_MEIPASS", "/tmp/pyinstaller-meipass", create=True):
                base_dir = get_base_dir()

        self.assertEqual(Path("/tmp/pyinstaller-meipass"), base_dir)

    def test_get_base_dir_uses_project_directory_when_not_frozen(self) -> None:
        with patch.object(rag_chatbot.sys, "frozen", False, create=True):
            base_dir = get_base_dir()

        self.assertEqual(Path(rag_chatbot.__file__).resolve().parent, base_dir)

    def test_load_corpus_splits_paragraphs_into_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            (corpus_dir / "agenda.txt").write_text(
                "Primeiro bloco.\n\nSegundo bloco.", encoding="utf-8"
            )

            chunks = load_corpus(corpus_dir)

        self.assertEqual(2, len(chunks))
        self.assertEqual("agenda.txt", chunks[0].source)
        self.assertEqual("Segundo bloco.", chunks[1].content)

    def test_keyword_retriever_prioritizes_matching_chunk(self) -> None:
        chunks = load_corpus(Path(__file__).resolve().parents[1] / "data")
        retriever = KeywordRetriever(chunks)

        results = retriever.search("Como chegar na Quinta da Boa Vista?", top_k=2)

        self.assertTrue(results)
        self.assertEqual("agenda_zona_norte.txt", results[0].source)

    def test_chatbot_answers_and_saves_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.jsonl"
            chatbot = FAQRAGChatbot(
                corpus_dir=Path(__file__).resolve().parents[1] / "data",
                history_path=history_path,
            )

            answer = chatbot.answer_question("Quais comidas típicas encontro nas festas?")

            self.assertIn("Fontes consultadas", answer)
            self.assertIn("canjica", answer)
            self.assertIn("dicas_visitante.txt", answer)
            self.assertNotIn("agenda_centro.txt", answer)
            history_entry = json.loads(history_path.read_text(encoding="utf-8").strip())
            self.assertEqual("Quais comidas típicas encontro nas festas?", history_entry["question"])

    def test_build_prompt_includes_contexts_and_question(self) -> None:
        retriever = KeywordRetriever(load_corpus(Path(__file__).resolve().parents[1] / "data"))
        retriever_contexts = retriever.search("horário Praça XV", top_k=1)
        prompt = build_prompt("Qual é o horário?", retriever_contexts)

        self.assertIn("Qual é o horário?", prompt)
        self.assertIn(retriever_contexts[0].content, prompt)


if __name__ == "__main__":
    unittest.main()
