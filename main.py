import streamlit as st
import tempfile

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from loaders import *
from rag import cria_vectorstore, formata_documentos_recuperados


TIPOS_ARQUIVOS = [
    'URL', 'YouTube', 'PDF', 'CSV', 'TXT'
]


MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'meta-llama/llama-4-scout-17b-16e-instruct'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-5-nano', 'gpt-4o-mini', 'gpt-5-mini', 'gpt-5.4-mini'],
        'chat': ChatOpenAI
    }
}

K_CHUNKS_RECUPERADOS = 6


def cria_historico():
    return []


def carrega_arquivo(tipo_arquivo, arquivo):
    documentos = []

    if tipo_arquivo == 'URL':
        documentos = upload_url(arquivo)
    elif tipo_arquivo == 'YouTube':
        documentos = upload_youtube(arquivo)
    elif tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            temp_name = temp.name
        documentos = upload_pdf(temp_name)
    elif tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            temp_name = temp.name
        documentos = upload_csv(temp_name)
    elif tipo_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            temp_name = temp.name
        documentos = upload_txt(temp_name)

    return documentos


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    if not api_key:
        st.error('Informe a API Key antes de inicializar o Orpheus.')
        st.stop()

    if not arquivo:
        st.error('Selecione ou faça upload de um arquivo antes de inicializar o Orpheus.')
        st.stop()

    with st.spinner('Carregando, dividindo em chunks e gerando embeddings...'):
        documentos = carrega_arquivo(tipo_arquivo, arquivo)

        if not documentos:
            st.error('Não foi possível extrair conteúdo do arquivo/fonte informado.')
            st.stop()


        vectorstore = cria_vectorstore(documentos, provedor, api_key)

        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': K_CHUNKS_RECUPERADOS, 'fetch_k': 20}
        )
        total_chunks = vectorstore.index.ntotal


    system_message = '''Você é um assistente amigável chamado Orpheus.

    Você utiliza uma técnica chamada RAG (Retrieval-Augmented Generation):
    em vez de receber o documento {tipo} inteiro, você recebe apenas os
    trechos mais relevantes para a pergunta atual do usuário, buscados por
    similaridade semântica.

    TRECHOS RECUPERADOS:
    ####
    {{context}}
    ####

    Responda com base apenas nas informações acima. Se a resposta não estiver
    nos trechos recuperados, diga claramente que não encontrou essa
    informação no documento carregado (ela pode existir no documento, mas não
    ter sido recuperada nessa busca) — não invente informações.

    Sempre que houver $ na sua saída, substitua por S.'''.format(tipo=tipo_arquivo)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    st.session_state['retriever'] = retriever
    st.session_state['historico'] = cria_historico()
    st.success(f'Orpheus inicializado! Documento dividido em {total_chunks} trechos indexados.')


# PÁGINA INICIAL ===================================================================

def pagina_chat():
    st.header('Bem-vindo ao Orpheus', divider=True)

    chain = st.session_state.get('chain')
    retriever = st.session_state.get('retriever')
    if chain is None or retriever is None:
        st.error('Carregue o Orpheus')
        st.stop()

    historico = st.session_state.get('historico')
    if historico is None:
        historico = cria_historico()
        st.session_state['historico'] = historico

    for mensagem in historico:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('fale com o orpheus')
    if input_usuario:

        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        docs_relevantes = retriever.invoke(input_usuario)
        contexto = formata_documentos_recuperados(docs_relevantes)

        chat = st.chat_message('ai')

        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': historico,
            'context': contexto
        }))


        if docs_relevantes:
            with chat.expander('📄 Fontes utilizadas nessa resposta'):
                for i, doc in enumerate(docs_relevantes, start=1):
                    fonte = doc.metadata.get('source', 'desconhecida')
                    pagina = doc.metadata.get('page')
                    referencia = fonte + (f' (página {pagina + 1})' if pagina is not None else '')
                    st.caption(f'**Trecho {i}** — {referencia}')
                    trecho_resumido = doc.page_content[:300]
                    if len(doc.page_content) > 300:
                        trecho_resumido += '...'
                    st.text(trecho_resumido)

        historico.append(HumanMessage(content=input_usuario))
        historico.append(AIMessage(content=resposta))
        st.session_state['historico'] = historico


# SIDEBAR ======================================================================

def sidebar():
    arquivo = None
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelo'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS)
        if tipo_arquivo == 'URL':
            arquivo = st.text_input('Digite a URL')
        if tipo_arquivo == 'YouTube':
            arquivo = st.text_input('Digite a URL ou o ID do vídeo')
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader('Faça o upload do arquivo', type='.pdf')
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo', type='.csv')
        if tipo_arquivo == 'TXT':
            arquivo = st.file_uploader('Faça o upload do arquivo', type='.txt')

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo', MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API Key para o {provedor}',
            value=st.session_state.get(f'api_key_{provedor}', ''),
            type='password')

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Orpheus'):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar histórico de conversa'):
        if st.session_state.get('chain') is None:
            st.warning('Inicialize o Orpheus antes de apagar o histórico.')
        else:
            st.session_state['historico'] = cria_historico()
            st.success('Histórico apagado.')


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
