import streamlit as st
import pickle
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the trained classifier and TfidfVectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def extract_text_from_pdf(pdf_file):
    """
    Function to extract text from a PDF file using PyPDF2 library.
    """
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
    show_pdf_file(text)
    return text

def show_pdf_file(pdf_text):
    file_container = st.expander("Your PDF file :")
    file_container.write(pdf_text)

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def main():
    st.title('Career Predictor App')
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf'])

    if uploaded_file is not None:
        try:
            # Extract text from uploaded PDF file
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error("Error occurred while extracting text from PDF: {}".format(e))
            return

        # Clean the resume text and transform it into the correct input format
        clean_resume = cleanResume(resume_text)
        input_features = tfidf.transform([clean_resume])

        # Make a prediction
        prediction_id = clf.predict(input_features)[0]

        # Define category mappings

        category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }
        
        # Get the predicted category name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display the predicted category name and text from PDF
        st.markdown(f"Based on your resume, your most suitable career could be a <span style='color:yellow'>**{category_name}**</span>", unsafe_allow_html=True)# Run the main function when the script is executed
if __name__ == '__main__':
    main()
