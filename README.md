#### Orpheus - Assistente com LangChain

<p align="center">
  <img src="assets/demo.gif" width="800"/>
</p>

<small>O Orpheus é uma aplicação construída com Streamlit que utiliza a arquitetura RAG (Geração Aumentada por Recuperação) para permitir que os usuários conversem com seus próprios dados. O projeto utiliza o LangChain para orquestrar o fluxo de dados e suporta integração com modelos da OpenAI e Groq, extraindo contexto de arquivos e links para gerar respostas fundamentadas.</small>

---

##### Funcionalidades
<small>
- Suporte multimodal: processamento de arquivos locais (PDF, CSV, TXT) e web (URLs e YouTube).
- Busca semântica: uso de FAISS e embeddings para recuperação de contexto relevante.
- Flexibilidade: escolha entre provedores Groq e OpenAI diretamente na interface.
- Transparência: exibição das fontes exatas utilizadas nas respostas.
- Interface: chat interativo e histórico de conversas.
</small>

##### Tecnologias
<small>Python, Streamlit, LangChain, FAISS, PyPDF, BeautifulSoup, YouTube Transcript API.</small>

---

##### Como rodar

<small>1. Clonar o repositório:</small>
```bash
git clone [https://seu-repositorio-aqui.git](https://seu-repositorio-aqui.git)
cd orpheus

<small>2. Criar e ativar ambiente virtual:</small>
```bash
python -m venv venv
venv\Scripts\activate

<small>3. Instalar dependências:</small>
```bash
pip install -r requirements.txt

<small>4. Rodar aplicação:</small>
```bash
streamlit run main.py