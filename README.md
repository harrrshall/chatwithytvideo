# YouTube Transcription Q&A Pipeline

This repository contains a Python script that allows you to retrieve the transcription of a YouTube video, split it into chunks, and answer questions related to the content of the video using OpenAI's GPT-3.5 model.

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.6 or later
- An OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
```

2. Navigate to the project directory:
```bash
cd your-repo
pip install -r requirements.txt
```



3. Open the script.py file and replace the following variables with your YouTube video URL and OpenAI API key:
python
``` bash
youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"
OPENAI_API_KEY = "your_openai_api_key"
PINECOPE_API_KEY= "pinecone_api_key"
Run the script:
```




The script will:
Download the transcription of the YouTube video (if it hasn't been downloaded already)
Split the transcription into chunks
Prompt you to enter a question related to the video content
Provide an answer using the GPT-3.5 model
License
This project is licensed under the MIT License.
