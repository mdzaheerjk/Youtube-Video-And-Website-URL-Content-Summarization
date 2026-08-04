import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title='Langchain: Summarize Text From YT or Website ', page_icon="🦜")
st.title("🦜 Langchain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# FIX 1: Changed st.slider to st.sidebar
with st.sidebar:
    groq_api_key = st.text_input("GROQ API Key", value="", type='password')

generic_url = st.text_input("URL", label_visibility="collapsed")

# FIX 2: Modern model name parameter (model_name -> model)
llm = ChatGroq(model='gemma-7b-it', groq_api_key=groq_api_key)

promt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""

# FIX 3: Fixed parameter name from input_variable to input_variables
prompt = PromptTemplate(template=promt_template, input_variables=['text'])

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please Provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can may be a YT Video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                if 'youtube.com' in generic_url or 'youtu.be' in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # FIX 4: Replaced load_summarize_chain with a modern LCEL chain pipeline
                combined_text = "\n".join([doc.page_content for doc in docs])
                chain = prompt | llm
                response = chain.invoke({"text": combined_text})

                # FIX 5: Output the text content cleanly
                st.success(response.content)
        except Exception as e:
            st.exception(f"Exception: {e}")
