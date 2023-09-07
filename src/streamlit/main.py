import sys
import os
import json.decoder 
import streamlit as st
import requests
import pandas as pd

from PIL import Image

def load_data():
    df1 = pd.read_csv('data/tourism_rating.csv')
    df2 = pd.read_csv('data/tourism_with_id.csv')
    df3 = pd.read_csv('data/user.csv')

    raw = pd.merge(df1, df3, on='User_Id')

    df = pd.merge(df2, raw, on='Place_Id').drop(columns=['Unnamed: 11', 'Unnamed: 12', 'Lat', 'Coordinate', 'Long', 'Time_Minutes'])

    return df

st.set_page_config(
    page_title="Tourist Recommendation System",
    page_icon=":bar_chart:",
    layout="centered"
)

img = Image.open("src/streamlit/assets/bg.jpg")
st.image(img)
st.title("Tourist Recommendation System")

GIVE_RECOMMENDATION_LABEL = "Give Recommendation"
SENDING_MESSAGES = "Sending data to the prediction server... please wait..."
INVALID_RESPONSE_MESSAGE = "Invalid response from the server. Please try again later."

metric_options = ['Top Rating Recommendations', 'Content Based Filtering', 'Collaborative Filtering']
selected_metric = st.selectbox('Select Method', metric_options)

if selected_metric == 'Top Rating Recommendations':

    with st.form(key="topk_recsys_form", clear_on_submit=True):
        k = st.text_input(
            label="Enter total recommendations:",
            help="default 10 recommendations"
        )

        submitted = st.form_submit_button(GIVE_RECOMMENDATION_LABEL)
    
        if submitted:
            # collect data from form
            form_data = {
                "k":k
            }

            with st.spinner(SENDING_MESSAGES):
                try:
                    res = requests.post("http://127.0.0.1:8081/topk", json=form_data).json()
                    
                    if res['status'] == 200:                        
                        recommendations = res["data"]
                        st.success("Top Place Recommendations:")
                        for rec in recommendations:
                            st.markdown(f"**Place Name:** {rec['Place_Name']}")

                            st.markdown(f"**Description:** {rec['Description']}")

                            st.markdown(f"**Category:** {rec['Category']}")

                            st.markdown(f"**City:** {rec['City']}")

                            st.markdown(f"**Price:** {rec['Price']}")

                            st.markdown(f"**Place Ratings:** {rec['Place_Ratings']}")

                            st.write("-" * 50)

                    else:
                        st.error(f"Error in prediction. Message: {res['message']}")
                except json.decoder.JSONDecodeError:
                    st.error(INVALID_RESPONSE_MESSAGE)

elif selected_metric == 'Content Based Filtering':

    df = load_data()
    unique_place_names = df['Place_Name'].unique()

    with st.form(key="contentbased_recsys_form", clear_on_submit=True):
        place_name = st.selectbox(
            label="Select a Place Name:",
            options=unique_place_names,
            help="Choose a place name from the dropdown"
        )

        total = st.text_input(
            label="Enter total recommendations:",
            help="default 10 recommendations"
        )

        submitted = st.form_submit_button(GIVE_RECOMMENDATION_LABEL)

        if submitted:
            # collect data from form
            form_data = {
                "place_name": place_name,
                "total":total
            }

            with st.spinner(SENDING_MESSAGES):
                try:
                    res = requests.post("http://127.0.0.1:8000/contentbased", json=form_data).json()
                    print(res)
                    if res['status'] == 200:
                        st.subheader("Profil Place")
                        user_input = res['user_input']
                        desc = res['user_input'][1]
                        cat = res['user_input'][2]
                        city = res['user_input'][3]
                        
                        st.markdown(f"**Place Name:** {place_name}")

                        st.markdown(f"**Place Description:** {desc['0']}")

                        st.markdown(f"**Place Category:** {cat['0']}")

                        st.markdown(f"**Place City:** {city['0']}")

                        recommendations = res['recommendations']
                        st.success(f"Recommended Places:")
                        for rec in recommendations:
                            st.markdown(f"**Place Name:** {rec['Place_Name']}")

                            st.markdown(f"**Description:** {rec['Description']}")

                            st.markdown(f"**Category:** {rec['Category']}")

                            st.markdown(f"**City:** {rec['City']}")

                            st.markdown(f"**Price:** {rec['Price']}")

                            st.write("-" * 50)
                    else:
                        st.error(f"Error in prediction. Please check your code: {res['message']}")
                except json.decoder.JSONDecodeError:
                    st.error(INVALID_RESPONSE_MESSAGE)


elif selected_metric == 'Collaborative Filtering':
    results = []

    with st.form(key="collaborative_recsys_form", clear_on_submit=True):
        user_id = st.text_input(
            label="Enter User ID:",
            help="Example value: 91"
        )

        submitted = st.form_submit_button(GIVE_RECOMMENDATION_LABEL)

        if submitted:
            # collect data from form
            form_data = {
                "user_id": user_id
            }

            with st.spinner(SENDING_MESSAGES):
                try:
                    res = requests.post("http://127.0.0.1:8080/collaborative", json=form_data).json()
                
                    if res['status'] == 200:
                        top_places_user = res['recommendations']['top_places_user']
                        recommended_places = res['recommendations']['recommended_places']

                        user_age = res['input'][1]
                        user_location = res['input'][2]

                        st.header(f"Profile UserID {user_id}:")

                        st.write(f"Age of user {user_age} years")
                        st.write(f"From {user_location}")

                        st.subheader(f"Top Places by UserID {user_id}:")
                        for place in top_places_user:
                            st.write(place)

                        st.success("Recommendations:")

                        st.subheader(f"Recommended Places for UserID {user_id}:")
                        for place in recommended_places:
                            st.write(place)
                        
                        results.append(res['recommendations']) 
                    else:
                        st.error(f"Error in prediction. Please check your code: {res['message']}")
                except json.decoder.JSONDecodeError:
                    st.error(INVALID_RESPONSE_MESSAGE)

