#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
BUILD_DIR="${PROJECT_ROOT}/build"

cd "${PROJECT_ROOT}"

python -m pip install --upgrade pip
python -m pip install pyinstaller

DATA_SEP=":"
if [[ "${OS:-}" == "Windows_NT" ]]; then
  DATA_SEP=";"
fi

pyinstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name faq_rag_chatbot \
  --add-data "data${DATA_SEP}data" \
  rag_chatbot.py

EXECUTABLE_NAME="faq_rag_chatbot"
if [[ "${OS:-}" == "Windows_NT" ]]; then
  EXECUTABLE_NAME="faq_rag_chatbot.exe"
fi

PACKAGE_DIR="${DIST_DIR}/faq_rag_chatbot_package"
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

cp "${DIST_DIR}/${EXECUTABLE_NAME}" "${PACKAGE_DIR}/"
cp -r "${PROJECT_ROOT}/data" "${PACKAGE_DIR}/data"

cat > "${PACKAGE_DIR}/README_EXECUTAVEL.txt" <<'EOF'
Como executar:
1) Abra o terminal na pasta do pacote.
2) Rode:
   - Linux/macOS: ./faq_rag_chatbot
   - Windows: faq_rag_chatbot.exe
3) Para pergunta única:
   - Linux/macOS: ./faq_rag_chatbot --question "Como chegar na Quinta da Boa Vista?"
   - Windows: faq_rag_chatbot.exe --question "Como chegar na Quinta da Boa Vista?"

Observações:
- O histórico é salvo em ~/.chatbot_faq_rag/chat_history.jsonl (ou no diretório definido por CHATBOT_FAQ_RAG_HOME).
- O executável deve ser gerado no mesmo sistema operacional de destino.
EOF

PLATFORM_NAME="windows"
if [[ "${OS:-}" != "Windows_NT" ]]; then
  if command -v uname >/dev/null 2>&1; then
    PLATFORM_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
  else
    PLATFORM_NAME="unknown"
  fi
fi

ZIP_NAME="faq_rag_chatbot_package_${PLATFORM_NAME}.zip"
(
  cd "${DIST_DIR}"
  rm -f "${ZIP_NAME}"
  zip -r "${ZIP_NAME}" "$(basename "${PACKAGE_DIR}")"
)

rm -rf "${BUILD_DIR}"
echo "Pacote gerado em: ${DIST_DIR}/${ZIP_NAME}"
