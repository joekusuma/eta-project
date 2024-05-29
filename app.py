import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp
import requests
import time
from backoff import on_exception, expo
from ratelimit import limits, RateLimitException
from openai import OpenAI
import openai
import os
from googleapiclient.discovery import build
from piazza_api import Piazza
from PIL import Image
from io import BytesIO
import pytesseract
from bs4 import BeautifulSoup
import html2text
from stackapi import StackAPI

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

DEFAULT_MODEL = "gpt-4"

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
def print_youtube_captions(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"Captions for video {video_id}:")
        for entry in transcript:
            timestamp = entry['start']
            text = entry['text']
            print(f"[{timestamp}] {text}")
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        print("No transcript found for this video.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def analyze_transcript_for_timestamp(video_id, query):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([f"[{entry['start']}] {entry['text']}" for entry in transcript])
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        client = openai.OpenAI(api_key=openai_api_key)

        messages = [{
            "role": "system",
            "content": "Your task is to find the timestamp (in seconds) within a video transcript where a specific topic is mentioned. Respond with the numeric timestamp only. \nYour single timestamp response must be integer ONLY"
        }, {
            "role": "user",
            "content": f"Please find the timestamp for when the topic '{query}' is discussed in the following transcript:\n{transcript_text}\nYour single timestamp response must be integer ONLY"
        }]

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", 
            messages=messages,
            stream=True   
        )
        
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
        
        timestamp = int(float(response_content))
        print(f"Timestamp: {timestamp}")
        return timestamp
    except (TranscriptsDisabled, NoTranscriptFound):
        print("Transcript not available for this video.")
    except ValueError:
        print("Failed to extract a numeric timestamp from the AI response.")
    except Exception as e:
        print(f"Error during transcript analysis: {e}")

    return None

def search_youtube_videos(query, max_results=2):
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results, 
        q=query,
        type="video"
    )
    response = request.execute()
    
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        print_youtube_captions(video_id)
        timestamp = analyze_transcript_for_timestamp(video_id, query)
        print(f"Timestamp2: {timestamp}")
        if timestamp is not None:
            timestamp_url = f"https://www.youtube.com/watch?v={video_id}&t={timestamp}s"
        else:
            timestamp_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Timestamp3: {timestamp_url}")
        
        description = generate_video_description(video_id, query)
        
        videos.append({
            'title': item['snippet']['title'],
            'videoId': video_id,
            'thumbnail': item['snippet']['thumbnails']['high']['url'],
            'timestamp_url': timestamp_url,
            'description': description
        })
        print(f"Video: {videos}")
    
    return videos

def generate_video_description(video_id, query):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        client = openai.OpenAI(api_key=openai_api_key)

        messages = [{
            "role": "system",
            "content": "Summarize the video in two sentences and explain how it relates to the specified topic."
        }, {
            "role": "user",
            "content": f"Summarize this video and explain how it relates to the topic '{query}': {transcript_text}"
        }]

        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        response_content = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content.append(chunk.choices[0].delta.content.strip())

        formatted_description = " ".join(response_content)
        print(f"Generated Description for {video_id}: {formatted_description}")
        return formatted_description
    except (TranscriptsDisabled, NoTranscriptFound):
        print("Transcript not available for this video.")
        return "No transcript available."
    except Exception as e:
        print(f"Error generating video description: {e}")
        return "An error occurred while generating the description."


def search_images(query, max_results=5):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not google_api_key or not cse_id:
        raise Exception("Google API key or Custom Search Engine ID not found. Please set the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")

    service = build("customsearch", "v1", developerKey=google_api_key)
    res = service.cse().list(
        q=query,
        cx=cse_id,
        searchType="image",
        num=max_results,
        key=google_api_key 
    ).execute()
    
    images = []
    if 'items' in res:
        for item in res['items']:
            images.append(item['link'])
    
    print(f"Image search query: {query}")  
    return images

def get_vectorstore(text_chunks):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_text_from_images(image_files):
    extracted_texts = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        extracted_texts += text + " "
    return extracted_texts

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or " "
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def load_images(image_urls):
    images = []
    for url in image_urls:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() 
            img = Image.open(BytesIO(response.content))
            images.append(img)
        except requests.RequestException as e:
            print(f"Request error for {url}: {e}")
        except IOError as e:
            print(f"Error opening image from {url}: {e}")
    return images


def display_multimedia_content(answer_text, show_images, show_videos):
    image_html = ""
    video_html = ""
    if show_images:
        image_urls = search_images(answer_text)
        if image_urls:
            images = load_images(image_urls[:2])  
            image_html += "<h3>Related Images</h3>"
            for index, image in enumerate(images):
                image_html += f"<img src='{image_urls[index]}' style='max-width: 100%; display: block; margin: 5px auto;'>"
    
    if show_videos:
        video_info = search_youtube_videos(answer_text, max_results=2)
        if video_info:
            video_html += "<h3>Related Videos</h3>"
            for video in video_info:
                video_html += f"<div style='margin-top: 10px;'><img src='{video['thumbnail']}' style='width: 100px; float: left; margin-right: 10px;'><a href='{video['timestamp_url']}' target='_blank'>{video['title']}</a><br><br><strong>Summary:</strong> <span>{video['description']}</span></div><div style='clear: both;'></div>"

    return image_html, video_html

def fetch_piazza_posts(network_id):
    piazza_email = os.getenv("PIAZZA_EMAIL")
    piazza_password = os.getenv("PIAZZA_PASSWORD")
    if not piazza_email or not piazza_password:
        raise Exception("Piazza credentials not found. Please set the PIAZZA_EMAIL and PIAZZA_PASSWORD environment variables.")
    
    p = Piazza()
    p.user_login(email=piazza_email, password=piazza_password)
    network = p.network(network_id)

    posts = []
    for post in network.iter_all_posts(limit=10):
        content = post['history'][0].get('content', 'No content available')
        posts.append(content)

    return posts

def integrate_piazza_with_chatbot(network_id):
    piazza_posts = fetch_piazza_posts(network_id)
    piazza_text_chunks = get_text_chunks(' '.join(piazza_posts))
    return piazza_text_chunks


def gpt_answer(question, reference_material=None, model=DEFAULT_MODEL, stack_overflow_results=None):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    client = openai.OpenAI(api_key=openai_api_key)
    messages = []
    if reference_material:
        messages.append({"role": "system", "content": reference_material})
    if stack_overflow_results:
        stack_info = ""
        for result in stack_overflow_results:
            stack_info += f"{result['question']}\n{result['answer']}\n"
        messages.append({"role": "system", "content": stack_info})

    messages.append({"role": "user", "content": question})
    
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_content += chunk.choices[0].delta.content
    
    return response_content

def handle_userinput(user_question, reference_material, model, show_images, show_videos, search_stack):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_question.lower() == 'exit':
        st.stop()
    else:
        response = ""
        stack_overflow_link = ""
        stack_overflow_result = None

        if search_stack and user_question:
            stack_overflow_results = search_stack_overflow(user_question)
            if stack_overflow_results:
                stack_overflow_result = stack_overflow_results[0]
                stack_overflow_link = stack_overflow_result['link']

            response = gpt_answer(user_question, reference_material, model, [stack_overflow_result] if stack_overflow_result else None)

            if stack_overflow_link:
                response += f"\nFor more detailed information, you can check out this [Stack Overflow post]({stack_overflow_link})."
        else:
            response = gpt_answer(user_question, reference_material, model)

        image_html, video_html = "", ""
        if show_images or show_videos:
            image_html, video_html = display_multimedia_content(response, show_images, show_videos)

        combined_response = f"{response}<br>{image_html if show_images else ''}{video_html if show_videos else ''}"
        st.session_state.chat_history.append({'speaker': 'bot', 'content': combined_response})
        st.session_state.chat_history.append({'speaker': 'user', 'content': user_question})

        for message in reversed(st.session_state.chat_history):
            if message['speaker'] == 'user':
                st.markdown(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

def handle_uploads():
    st.subheader("Upload")
    uploaded_files = st.file_uploader("Upload your documents or images", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])
    text_content = ""
    if uploaded_files:
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text_from_pdf = get_pdf_text([uploaded_file])
                    if text_from_pdf.strip():  
                        text_content += f"[PDF Content]\n{text_from_pdf}\n\n"
                else:
                    text_from_image = get_text_from_images([uploaded_file])
                    if text_from_image.strip():  
                        text_content += f"[Image Content]\n{text_from_image}\n\n"
            st.success("Content processed. You can now ask questions.")
    return text_content

def preprocess_stack_overflow_answer(html_content):
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.bypass_tables = False
    text_maker.ignore_images = True
    text_maker.ignore_emphasis = True
    plain_text = text_maker.handle(html_content)

    plain_text = plain_text.strip()

    print("Debug: Plain text after HTML stripping:", plain_text)
    
    return plain_text

def search_stack_overflow(query):
    SITE = StackAPI('stackoverflow')

    try:
        questions = SITE.fetch('search/advanced', q=query, sort='relevance', 
                               order='desc', filter='!-*jbN-o8P3E5')
        
        relevant_questions = [q for q in questions['items'] if q.get('is_answered', False)][:5]
        
        results = []
        for item in relevant_questions:
            if 'accepted_answer_id' in item:
                answer_id = item['accepted_answer_id']
            else:
                answers = item.get('answers', [])
                if answers:
                    answers = sorted(answers, key=lambda x: x['score'], reverse=True)
                    answer_id = answers[0]['answer_id']
                else:
                    continue

            answer = SITE.fetch('answers', ids=[answer_id], filter='withbody')

            if answer['items']:
                html_content = answer['items'][0]['body']
                answer_text = preprocess_stack_overflow_answer(html_content)
            else:
                answer_text = "Answer not available."

            result = {
                "question": item['title'],
                "answer": answer_text,
                "link": item['link']
            }
            results.append(result)
        
        return results

    except Exception as e:
        print(f"Error fetching from Stack Overflow: {e}")
        return []

def main():
    st.set_page_config(page_title="Electronic TA", page_icon="🎭")
    st.markdown(css, unsafe_allow_html=True)
    st.header("Electronic TA 🎭")

    with st.sidebar:
        st.subheader("Settings")
        model_choice = st.selectbox("Choose GPT Model", ["gpt-4", "gpt-3.5-turbo-0125"], index=0)
        show_images = st.checkbox("Show Related Images", value=False)
        show_videos = st.checkbox("Show Related Videos", value=False)
        search_stack = st.checkbox("Search Stack Overflow", value=False)
        reference_material = handle_uploads()
        st.subheader("Piazza")
        use_piazza = st.checkbox("Use Piazza", value=False)
        network_id = ""
        if use_piazza:
            st.subheader("Enter your Piazza Class Network ID (Optional)")
            network_id = st.text_input("Network ID:", value="lr7e73kounllq", key="network_id")
            if network_id:
                try:
                    with st.spinner("Fetching Piazza posts..."):
                        piazza_text = ' '.join(integrate_piazza_with_chatbot(network_id))
                        if piazza_text:
                            reference_material += "\n\n" + piazza_text  
                            st.success("Piazza content processed. You can now ask questions.")
                except Exception as e:
                    st.error(f"Failed to fetch Piazza content: {e}")

    user_question = st.text_input("Ask questions about coursework or uploaded documents:", key="user_question")
    if user_question:
        handle_userinput(user_question, reference_material, model_choice, show_images, show_videos, search_stack)

if __name__ == '__main__':
    main()
