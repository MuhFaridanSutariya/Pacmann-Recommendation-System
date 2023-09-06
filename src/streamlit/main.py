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

# Define constants
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
                    # Make a request to your FastAPI endpoint
                    res = requests.post("http://127.0.0.1:8081/topk", json=form_data).json()
                    
                    if res['status'] == 200:
                        st.success("Top K recommendations retrieved successfully.")
                        
                        # Display the recommendations
                        recommendations = res["data"]
                        st.write("Top K Recommendations:")
                        for rec in recommendations:
                            st.write(f"Place Name: {rec['Place_Name']}")
                            st.write(f"Description: {rec['Description']}")
                            st.write(f"Category: {rec['Category']}")
                            st.write(f"City: {rec['City']}")
                            st.write(f"Price: {rec['Price']}")
                            st.write(f"Place Ratings: {rec['Place_Ratings']}")
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

                    if res['status'] == 200:
                        st.success(f"User Input: {place_name}")
                        recommendations = res['recommendations']
                        st.write("Recommended Places:")
                        for rec in recommendations:
                            st.write(f"Place Name: {rec['Place_Name']}")
                            st.write(f"Description: {rec['Description']}")
                            st.write(f"Category: {rec['Category']}")
                            st.write(f"City: {rec['City']}")
                            st.write(f"Price: {rec['Price']}")
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

                        st.header("Profile UserID {}:".format(user_id))

                        st.write("Age of user {} years".format(user_age))
                        st.write("From {}".format(user_location))

                        st.subheader("Top Places by UserID {}:".format(user_id))
                        for place in top_places_user:
                            st.write(place)

                        st.success("Recommendations:")

                        st.subheader("Recommended Places for UserID {}:".format(user_id))
                        for place in recommended_places:
                            st.write(place)
                        
                        results.append(res['recommendations']) 
                    else:
                        st.error(f"Error in prediction. Please check your code: {res['message']}")
                except json.decoder.JSONDecodeError:
                    st.error(INVALID_RESPONSE_MESSAGE)

