import streamlit as st
import time



app_mode = st.sidebar.selectbox('Select Page',['HOME','TEXT TO IMAGE']) 
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

if app_mode=='HOME' :
    add_bg_from_local('/home/amine/Desktop/XD.png')    
elif app_mode=='TEXT TO IMAGE' :
    st.image("/home/amine/Desktop/blank.png")
    title = st.text_input("𝗪𝗛𝗔𝗧'𝗦 𝗜𝗡 𝗬𝗢𝗨𝗥 𝗠𝗜𝗡𝗗 ❓" ,)
    prompts = [titles]
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
        st.image(img)
        
    
    
    
