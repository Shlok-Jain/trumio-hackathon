import flask
from flask import request
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import time
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
import os
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import textwrap

GOOGLE_API_KEY='API KEY HERE'
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-1.5-flash')


nltk.download('punkt')

app = flask.Flask(__name__,static_folder='resume',static_url_path='/')
spacy_nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

dir_list = os.listdir("resume")
resumes = []
for name in dir_list:
    file_path = f"resume/{name}"
    pdf = PyPDF2.PdfReader(file_path)
    resume = ""
    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()
    resumes.append(resume)

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    #keep only alphabets
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

cs_model = Doc2Vec.load('cv_job_maching1.model')
def cosine_similarity1(jd):
    input_JD = preprocess_text(jd)
    similarities = []
    for resume in resumes:
        input_CV = preprocess_text(resume)
        v1 = cs_model.infer_vector(input_CV.split())
        v2 = cs_model.infer_vector(input_JD.split())
        similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
        similarities.append(similarity)
    res = (sorted(range(len(similarities)), key = lambda sub: similarities[sub])[-5:])
    res.reverse()
    student_names = []
    for index in res:
        student_names.append(dir_list[index])
    return student_names


def lsi_matching(ps, resumes):
  corpus = ps + resumes

  # Create TF-IDF vectorizer
  vectorizer = TfidfVectorizer(stop_words='english')

  # Fit the vectorizer to the corpus
  tfidf_matrix = vectorizer.fit_transform(corpus)

  # Apply Truncated SVD for dimensionality reduction (choose an appropriate number of topics)
  lsa = TruncatedSVD(n_components=50)  # Adjust n_components based on your data

  # Transform the TF-IDF matrix
  lsa_matrix = lsa.fit_transform(tfidf_matrix)

  # Get the LSA representation of the first problem statement (ps[0])
  ps_lsa = lsa_matrix[0]

  # Calculate cosine similarity between ps_lsa and each resume's LSA representation
  similarities = []
  for resume_lsa in lsa_matrix[len(ps):]:
    similarity = np.dot(ps_lsa, resume_lsa) / (np.linalg.norm(ps_lsa) * np.linalg.norm(resume_lsa))
    similarities.append(similarity)
    #get top 5 matches
  res = (sorted(range(len(similarities)), key = lambda sub: similarities[sub])[-5:])
  res.reverse()
  student_names = []
  for index in res:
    student_names.append(dir_list[index])
  return student_names

def keyword_matching_through_cosine_similarity(project_text, resumes_csv='resumes.csv'):
    resumes_df = pd.read_csv('resumes.csv')   
    
    # project_text_preprocessed = preprocess_text(project_text)
    # Pre-processing
    project_text = re.sub(r'http\S+|www\S+', '', project_text, flags=re.MULTILINE)
    project_text = re.sub(r'[^a-zA-Z\s]', '', project_text)
    project_text = project_text.lower()
    
    # project_keywords = extract_keywords(project_text_preprocessed)
    # Keyword Extraction
    doc = spacy_nlp(project_text)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    project_text = ' '.join(keywords)

    vectorizer = TfidfVectorizer()
    resume_vectors = vectorizer.fit_transform(resumes_df['keywords'])
    project_vector = vectorizer.transform([project_text])
    similarity_scores = cosine_similarity(project_vector, resume_vectors)[0]
    resumes_df['Similarity'] = similarity_scores
    top_resume_indices = similarity_scores.argsort()[-5:][::-1]
    top_resumes = resumes_df.iloc[top_resume_indices]
    return top_resumes['CANDIDATE_NAME'].tolist()

def get_gemini_repsonse(input, model):
    response=model.generate_content(input)
    return response

def preprocess_llm(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

def extract_pd_match(data):
    match = re.search(r'"PD Match":\s*"(\d+)%"', data)
    if match:
        return int(match.group(1))
    return None

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def gemini(input_pd):
    pd = preprocess_llm(input_pd)
    similarities = []
    for resume in resumes:
        text = preprocess_llm(resume)
        # Prompt Template
        model = genai.GenerativeModel('gemini-1.5-flash')

        input_prompt = f"""
        Hey Act Like a skilled or very experience ATS(Application Tracking System)
        with a deep understanding of tech field,software engineering,data science ,data analyst, big data engineer, consulting, finance, fintech, quant trading, web developer. Your task is to evaluate the resume based on the given project description.
        You must consider the project market is very competitive. Assign the percentage Matching based
        on project description (pd), similar projects the candidate has done, similar academic degree and similar skills. Give special weightage
        to similar internships the person has done, those are the most important. Also look at relevant courses he has done. Give special weightage to
        relevant competitions/conferences he has won/participated in. These should be relevant to the project description.

        Note : Do not give positions of responsibility and extracurricular activities much importance.

        Also give a profile summary and why you think the candidate is suitable for the project.

        Explicitly mention the details of only one
        project/internship/competition/conference experience of candiddate that is most relevant to the given project description (pd).
        Mention name of the project explicitly.
        Also mention possible weaknesses (not mandatory). While generating this pd match, this project/internship experience should be given the most weightage.

        MOST IMPORTANT : Make all your decisions based on the given current resume (text) and description (pd) only, not on the previous history.
        I don't want details of previous candidate or project, forget those!!

        resume:{text}
        description:{pd}

        I want the response as per below structure
        {{"PD Match": "%",
        "Profile Summary": "",
        "Most relevant project/internship experience":""}}

        Also store this pd match in a variable pd_match.
        """

        response = get_gemini_repsonse(input_prompt, model)
        print(to_markdown(response.text))
        pd_match = extract_pd_match(response.text)
        similarities.append(pd_match)

        
        time.sleep(60/15)

    res = (sorted(range(len(similarities)), key=lambda sub: similarities[sub])[-5:])
    res.reverse()
    student_names = []
    for index in res:
        student_names.append(dir_list[index])
    return student_names

@app.route('/')
def index():
    #return index.html
    return flask.render_template('index.html')

resume_cleaned = [preprocess_text(resume) for resume in resumes]
@app.route('/recommend', methods=['POST'])
def getresume():
    data = request.get_json()
    jd = data['jd']
    methodr = data['method']
    if methodr == 'cs':
        result = cosine_similarity1(jd)
        return flask.jsonify({'top5': result})
        # pass
    elif methodr == 'lsa':
        jd_cleaned = preprocess_text(jd)
        top5 = lsi_matching([jd_cleaned], resume_cleaned)
        return flask.jsonify({'top5': top5})
    elif methodr == 'keyword':
        result = keyword_matching_through_cosine_similarity(jd)
        return flask.jsonify({'top5': result})
    elif methodr == 'llm':
        result = gemini(jd)
        return flask.jsonify({'top5': result})


        

if __name__ == '__main__':
    app.run(port=5000, debug=True)