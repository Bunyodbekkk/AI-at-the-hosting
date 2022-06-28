import streamlit as st
from fastai.vision.all import *

import pathlib
import plotly.express as px             
import platform
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
plt=platform.system()
if plt =="Linux":pathlib.WindowsPath==pathlib.PosixPath


st.title('BED,COUCH,TABLElarni klassifikatsiya qiluvchi model')     
file=st.file_uploader('RASM YUKLASHINGIZ MUMKIN',type=['png','jpg','jpeg','jfif','gif','svg'])
if file:
 st.image(file)
 img=PILImage.create(file)
 model=load_learner('AI_model.pkl')

 pred, pred_id, probs=model.predict(img)
 st.success(f'Bashorat: {pred}')
 st.info(f'Ehtimollik:{probs[pred_id]*100:.1f}')



 fig=px.bar(x=probs*100,y=model.dls.vocab)
 st.plotly_chart(fig)
