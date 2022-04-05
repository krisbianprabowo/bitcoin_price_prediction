import streamlit as st

#-----------twitter information----------#
st.title('Twitter Information')

col1, col2 = st.columns(2)

with col1:
    st.number_input('Insert a number of followers')

with col2:
    st.selectbox('Is a Account is verified or not',('Yes','No'))

st.text_input('Input Tweets')

#------------Inputasi Price-----------#
st.title('Price Information')

op, cp= st.columns(2)

with op:
    st.number_input('Open Price')
with cp:
    st.number_input('Close Price')


hp, lp, vl = st.columns(3)

with hp:
    st.number_input('High Price')
with lp:
    st.number_input('Low Price')
with vl:
    st.number_input('volume')



#-----------price information------------#
# st.title('Price Action ')

# col1, col2 = st.columns(2)
# col1.metric('Open Price','1000','1200')
# col2.metric('Close Price','2000','-1000')

# col1, col2, col3 = st.columns(3)
# col1.metric('High',10) 
# col2.metric('Low',10) 
# col3.metric('Volume',10) 
