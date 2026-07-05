"""
Módulo responsável pela parte de RAG (Retrieval-Augmented Generation) do projeto.

Pipeline:
1) CHUNKING   -> quebra os documentos carregados em pedaços menores (chunks),
                 com sobreposição entre eles, para preservar contexto nas bordas.
2) EMBEDDING  -> transforma cada chunk em um vetor numérico que representa
                 seu significado semântico.
3) INDEXAÇÃO  -> guarda esses vetores num vector store (FAISS) que permite
                 busca por similaridade (não por palavra-chave).
4) RETRIEVAL  -> na hora da pergunta, busca os chunks mais parecidos com a
                 pergunta do usuário, em vez de mandar o documento inteiro
                 para o LLM.
"""



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings

MODELO_EMBEDDING_LOCAL = "intfloat/multilingual-e5-small"

MODELO_EMBEDDING_OPENAI = "text-embedding-3-small"


def escolhe_embeddings(provedor, api_key):

    if provedor == 'OpenAI':
        return OpenAIEmbeddings(model=MODELO_EMBEDDING_OPENAI, api_key=api_key)

    return FastEmbedEmbeddings(model_name=MODELO_EMBEDDING_LOCAL, doc_embed_type='passage')


def divide_documentos(documentos, chunk_size=1000, chunk_overlap=200):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documentos)


def cria_vectorstore(documentos, provedor, api_key):

    chunks = divide_documentos(documentos)
    embeddings = escolhe_embeddings(provedor, api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def formata_documentos_recuperados(docs):

    partes = []
    for i, doc in enumerate(docs, start=1):
        fonte = doc.metadata.get('source', 'desconhecida')
        pagina = doc.metadata.get('page')
        referencia = fonte + (f', página {pagina + 1}' if pagina is not None else '')
        partes.append(f'[Trecho {i} | fonte: {referencia}]\n{doc.page_content}')
    return '\n\n'.join(partes)
