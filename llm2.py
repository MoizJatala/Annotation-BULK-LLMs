import textwrap
import re
import pandas as pd
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

generated_responses = []
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove hashtags and mentions
    text = re.sub(r'#\w+|\@\w+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text


genai.configure(api_key="AIzaSyA6mrb7KgEsW-JuYxO2s43hIhBwNwVMGFo")
model = genai.GenerativeModel('gemini-pro')

sentences = pd.read_csv("Corona_NLP_test.csv",nrows=15)['OriginalTweet']

for sentence in sentences:
    cleaned_text = clean_text(sentence)
    ####Prompt 1
    response = model.generate_content(f"{cleaned_text} can you annotate according to the labels Positive, Extremely positive, Negative, Extremely Negative and give reason of annotation with headings of label and reason (only highest priority label and reason)")

####Prompt 2
    #response = model.generate_content(f"Imagine you are personally reacting to the text: '{cleaned_text}'. Annotate the sentiment as Positive, Extremely positive, Negative, or Extremely Negative. Give reasons for your annotations, emphasizing the highest priority label and reason.")
####Prompt 3   
   #response = model.generate_content(f"Consider the context and sentiment conveyed in the text: '{cleaned_text}'. Annotate it with labels Positive, Extremely positive, Negative, Extremely Negative. Provide detailed reasons for your annotations, focusing on the highest priority label and reason.")

    try:
       response.text
       generated_responses.append(response.text)
    except Exception as e:
       print(f'{type(e).__name__}: {e}')

# Create a DataFrame with the original sentences and generated responses
result_df = pd.DataFrame({'OriginalSentence': sentences, 'GeneratedResponse': generated_responses})

# Save the DataFrame to a CSV file
result_df.to_csv('generated_responses.csv', index=False)
