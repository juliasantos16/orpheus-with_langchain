import os
import re
from time import sleep

import requests
import streamlit as st
from fake_useragent import UserAgent

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_community.document_loaders import (WebBaseLoader,
                                                    CSVLoader,
                                                    PyPDFLoader,
                                                    TextLoader)


def extrai_video_id(url_ou_id):

    padroes = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',   
        r'youtu\.be\/([0-9A-Za-z_-]{11})',    
    ]
    for padrao in padroes:
        match = re.search(padrao, url_ou_id)
        if match:
            return match.group(1)

    return url_ou_id.strip()




def upload_url(url):
    documentos = []
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            documentos = loader.load()
            break
        except Exception as e:
            print(f'Erro ao carregar o site (tentativa {i + 1}/5): {e}')
            sleep(3)
    if not documentos:
        st.error('Não foi possível carregar o site.')
    return documentos


def obtem_titulo_youtube(video_id):

    try:
        url = f'https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json'
        resposta = requests.get(url, timeout=5)
        resposta.raise_for_status()
        return resposta.json().get('title')
    except Exception:
        return None


def upload_youtube(url_ou_id):
    video_id = extrai_video_id(url_ou_id)
    documentos = []
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["pt", "pt-BR", "en"])
        texto = "\n\n".join([snippet.text for snippet in transcript])

        titulo = obtem_titulo_youtube(video_id)
        if titulo:

            texto = f'Título do vídeo: {titulo}\n\n{texto}'
            fonte = f'youtube:{video_id} - {titulo}'
        else:
            fonte = f'youtube:{video_id}'

        documentos = [Document(page_content=texto, metadata={"source": fonte})]
    except Exception as e:
        st.error(f'Não foi possível obter a transcrição do vídeo: {e}')
    return documentos


def upload_csv(caminho):
    loader = CSVLoader(caminho)
    return loader.load()


def upload_pdf(caminho):
    loader = PyPDFLoader(caminho)
    return loader.load()


def upload_txt(caminho):
    loader = TextLoader(caminho)
    return loader.load()
