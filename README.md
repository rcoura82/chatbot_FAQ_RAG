# chatbot_FAQ_RAG

Chatbot de Perguntas Frequentes usando RAG simples com documentos locais sobre o Guia de Festas Juninas do RJ.

## O que foi implementado

- Corpus local em `data/` com arquivos `.txt`
- Pipeline RAG com:
  - indexação vetorial com `FAISS` + `sentence-transformers/all-MiniLM-L6-v2` quando as dependências estão instaladas
  - fallback local por palavras-chave para permitir execução básica sem downloads
  - geração de resposta com `google/flan-t5-base` quando disponível
- Interface de terminal em `rag_chatbot.py`
- Histórico de perguntas e respostas em `chat_history.jsonl`

## Instalação

```bash
cd <diretorio-do-projeto>
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

## Validação do destino de entrega

- **NotebookLM não executa binários** (`.exe`, ELF, app).
- Para **rodar o chatbot**, a entrega correta é um executável local (gerado por sistema-alvo).
- Para **usar no NotebookLM**, a entrega correta são os documentos-fonte (`.txt`, `.md`, `.pdf` etc).

## Empacotamento (executável local com PyInstaller)

> Gere o executável no mesmo sistema operacional em que ele será usado.

1. No projeto:

   ```bash
   cd /tmp/workspace/rcoura82/chatbot_FAQ_RAG
   chmod +x /tmp/workspace/rcoura82/chatbot_FAQ_RAG/scripts/package_executable.sh
   /tmp/workspace/rcoura82/chatbot_FAQ_RAG/scripts/package_executable.sh
   ```

2. Artefatos gerados:
   - Executável em `dist/`
   - Pacote pronto em `dist/faq_rag_chatbot_package_<so>.zip`
   - Pacote inclui executável + pasta `data/` + `README_EXECUTAVEL.txt`

3. Histórico de conversa:
   - Salvo fora da pasta de instalação em `~/.chatbot_faq_rag/chat_history.jsonl`
   - Opcional: defina `CHATBOT_FAQ_RAG_HOME` para customizar o diretório de saída.

## Entrega para NotebookLM (fontes de conteúdo)

Para preparar documentos prontos para upload:

```bash
cd /tmp/workspace/rcoura82/chatbot_FAQ_RAG
python /tmp/workspace/rcoura82/chatbot_FAQ_RAG/scripts/prepare_notebooklm_sources.py
```

Saída:
- Pasta `/tmp/workspace/rcoura82/chatbot_FAQ_RAG/notebooklm_sources/` com arquivos `.md` derivados de `data/*.txt`
- `README.md` com índice dos arquivos gerados

## Limitação do NotebookLM

- O NotebookLM **não roda executáveis** nem aplicações Python diretamente.
- Se precisar do chatbot rodando, hospede o app externamente (API/app web) e use o NotebookLM apenas para consulta às fontes.

## Checklist de validação da entrega

- [ ] Executável inicia em terminal no sistema-alvo
- [ ] Responde pelo menos uma pergunta de teste
- [ ] Mostra seção `Fontes consultadas`
- [ ] Encerra corretamente com `sair`
- [ ] Pacote zip contém executável + `data/` + instruções rápidas
