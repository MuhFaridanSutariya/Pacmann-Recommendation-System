from PIL import Image

import os
import streamlit as st
import requests


st.set_page_config(
    page_title="Tourist Recommendation System",
    page_icon=":bar_chart:",
    layout="centered"
)

img = Image.open("src/streamlit/assets/bg.jpg")
st.image(img)
st.title("Tourist Recommendation System")

results = []

with st.form(key="collaborative_recsys_form", clear_on_submit=True):
    user = st.text_input(
        label="Enter User ID:",
        help="Example value: 91"
    )

    submitted = st.form_submit_button("Give Recommendation")

    if submitted:
        # collect data from form
        form_data = {
            "user": user
        }

        with st.spinner("Sending data to the prediction server... please wait..."):
            res = requests.post("http://127.0.0.1:8000/recommendation", json=form_data).json()

        if res['status'] == 200:
            top_places_user = res['recommendations']['top_places_user']
            recommended_places = res['recommendations']['recommended_places']

            st.success("Recommendations:")
            st.subheader("Top Places UserID {}:".format(user))
            for place in top_places_user:
                st.write(place)

            st.subheader("Recommended Places for UserID {}:".format(user))
            for place in recommended_places:
                st.write(place)
            
            results.append(res['recommendations']) 
        else:
            st.error(f"Error in prediction. Please check your code: {res['message']}")


