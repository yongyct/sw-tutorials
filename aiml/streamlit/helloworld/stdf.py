import streamlit as st
import numpy as np
import pandas as pd

#df = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])

#st.dataframe(df.style.highlight_max(axis=0))
#st.table(df)
#st.line_chart(df)

def get_preds(x1, x2):
  return x1 + x2 - x1 * np.random.randn(1)[0] - x2 * np.random.randn(1)[0]

x1 = st.slider('x1')
x2 = 0 
st.number_input("x2", key="x2")

# You can access the value at any point with:
st.write(f'Feature values = ({x1}, {st.session_state.x2})')
st.write(f'Prediction = {get_preds(x1, st.session_state.x2)}')


