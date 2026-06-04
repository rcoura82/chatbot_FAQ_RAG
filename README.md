# chatbot_FAQ_RAG

Chatbot de Perguntas Frequentes usando RAG simples com documentos locais sobre o Guia de Festas Juninas do RJ.

## O que foi implementado

- Corpus local em `/tmp/workspace/rcoura82/chatbot_FAQ_RAG/data` com arquivos `.txt`
- Pipeline RAG com:
  - indexação vetorial com `FAISS` + `sentence-transformers/all-MiniLM-L6-v2` quando as dependências estão instaladas
  - fallback local por palavras-chave para permitir execução básica sem downloads
  - geração de resposta com `google/flan-t5-base` quando disponível
- Interface de terminal em `/tmp/workspace/rcoura82/chatbot_FAQ_RAG/rag_chatbot.py`
- Histórico de perguntas e respostas em `chat_history.jsonl`

## Instalação

```bash
cd /tmp/workspace/rcoura82/chatbot_FAQ_RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Pergunta única:

```bash
python rag_chatbot.py --question "Como chegar na Quinta da Boa Vista?"
```

Modo interativo:

```bash
python rag_chatbot.py
```

Digite `sair` para encerrar a conversa.
