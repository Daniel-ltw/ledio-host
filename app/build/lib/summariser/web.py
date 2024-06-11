from langchain.document_loaders import WebBaseLoader
from summariser.text import summarise_text
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

import re

def summarise_web(url):
    if 'youtube.com' in url:
        video_id = re.search(r'v=([^&]+)', url).group(1)
        summarise_youtube(video_id)
    else:
        loader = WebBaseLoader(url)
        pages = loader.load()
        return summarise_text(pages[0].page_content)

def summarise_youtube(id):
    transcript = YouTubeTranscriptApi.get_transcript(id)
    formatter = TextFormatter()
    text = formatter.format_transcript(transcript)
    return summarise_text(text)
