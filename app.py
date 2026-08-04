import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain  # Fixed this line
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title='Langchain: Summarize Text From YT or Website ',page_icon="🦜")
st.title("🦜 Langchian: Summarize Text From YT or Website")
st.subheader("Summarize URL")


with st.slider:
    groq_api_key=st.text_input("GROQ API Key",value="",type='password')

generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model='gemma-7b-It',groq_api_key=groq_api_key)

promt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=promt_template,input_variable=['text'])

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please Provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can may be a YT Video utl or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                if 'youtube.com' in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                chain=load_summarize_chain(llm,chain_type='stuff',prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")