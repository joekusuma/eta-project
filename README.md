# eTA (Electronic Teaching Assistant)

## Overview
eTA (Electronic Teaching Assistant) is an interactive application designed to enhance learning by providing personalized assistance. Using advanced AI models like OpenAI’s GPT and LangChain, eTA allows users to upload course materials, ask questions, and receive detailed responses enriched with multimedia content.

## Features
- **Interactive Q&A**: Engage in a conversational manner for coursework-related questions.
- **Document & Image Processing**: Upload PDFs and images; the app extracts text using OCR for analysis.
- **Multimedia Integration**:
  - Fetches related images via Google’s Custom Search API.
  - Searches YouTube for relevant videos, extracts transcripts, and provides timestamped links.
- **External Resource Support**:
  - **Piazza Integration**: Retrieves class discussions.
  - **Stack Overflow Search**: Pulls relevant technical Q&A.
- **Customizable Settings**:
  - Choose between GPT-4 and GPT-3.5-Turbo.
  - Enable or disable related images, videos, and Stack Overflow search.

## How It Works
1. **User Interaction**: Upload documents or images, then ask any question.
2. **Processing**:
   - Extracts text from uploads.
   - Retrieves context from external sources.
3. **Query Handling**:
   - Uses OpenAI’s GPT models to generate answers.
   - Enriches responses with images, videos, and external resources.
4. **Response Generation**: Presents detailed answers in an interactive chat interface.

## Technologies Used
- **Python**
- **Frameworks & Libraries**:
  - Streamlit (Web UI)
  - LangChain (LLM management)
  - PyPDF2 & Pytesseract (Text extraction)
  - OpenAI API (AI responses)
  - Google APIs (Custom Search, YouTube Data)
  - Piazza API (Class discussions)
  - StackAPI (Stack Overflow queries)
  - FAISS (Vector storage)

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project directory and add your API keys:

```
OPENAI_API_KEY=your_key_here
YOUTUBE_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_cse_id_here
PIAZZA_EMAIL=your_email
PIAZZA_PASSWORD=your_password
```

### 4. Run the application

```bash
streamlit run app.py
```
