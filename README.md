# Resume recommendation system using NLP

We have created a resume recommendation system that recommends resumes based upon the problem statement given by the recruiter. We have implemented 5 different approaches for this.
1. Doc2Vec model followed by cosine similarity
2. Using keyword matching
3. Using Gemini API
4. LSA matching
5. Using Llama3 LLM

We have implemented the server backend in flask. And frontend using HTML, CSS and JS.
Exact details of the methods can be found in the report.

# Running the code locally

1. Clone the repository
2. Install the required libraries using the following command:
```bash
pip install flask
pip install gensim
pip install nltk
pip install numpy
pip install termcolor
pip install pandas
pip install PyPDF2
pip install re
pip install plotly
pip install nltk spacy sklearn
pip install google-generativeai
pip install textwrap
```
3. Run the following command to start the server:
```bash
python app.py
```
4. Open the link displayed in the terminal.

# Note
- You will have to add your Gemini API key in the `app.py line 26` to use the Gemini API method.
- The resumes.csv was generated using `generate_csv.ipynb` file, it's code can be found there.
- The `cv_job_matching1.model` is created using `generate_cosine_model.ipynb`
