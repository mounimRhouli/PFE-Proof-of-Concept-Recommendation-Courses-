import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Function to load content-based data
def load_content_data():
    content_data = pd.read_csv('content_data.csv')
    content_data['description'] = content_data['description'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_data['description'])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return content_data, tfidf_vectorizer, content_similarity

# Function to get content-based recommendations
def content_based_recommend(course, content_data, tfidf_vectorizer, content_similarity):
    column_name = 'course_nam'
    if content_data is not None and column_name in content_data.columns:
        index = content_data[content_data[column_name] == course].index
        if not index.empty:
            index = index[0]
            distances_content = sorted(enumerate(content_similarity[index]), reverse=True, key=lambda x: x[1])
            recommended_course_names_content = [content_data.iloc[i[0]][column_name] for i in distances_content[1:7]]
            return recommended_course_names_content
    return []

# Function to load collaborative data
def load_collaborative_data():
    collaborative_data = pd.read_csv('collaborative_data.csv')
    return collaborative_data

# Function to get collaborative recommendations
def get_user_collaborative_recommendations(selected_user, collaborative_data, content_data, top_n=5):
    user_ratings = collaborative_data[collaborative_data['user_id'] == selected_user][['course_id', 'rating']]
    
    if user_ratings.empty:
        return []  # The selected user has not rated any courses
    
    # Find courses rated positively by users who have rated at least one course similarly to the selected user
    similar_users = collaborative_data[
        (collaborative_data['course_id'].isin(user_ratings['course_id'])) & 
        (collaborative_data['rating'] >= 4) &
        (collaborative_data['user_id'] != selected_user)
    ]['user_id'].unique().tolist()
    
    if not similar_users:
        return []  # No similar users found
    
    # Fetch course_ids from collaborative_data based on collaborative recommendations
    collaborative_recommendations = collaborative_data[
        (collaborative_data['user_id'].isin(similar_users)) & 
        (collaborative_data['rating'] >= 4)
    ]['course_id'].unique().tolist()
    
    if not collaborative_recommendations:
        return []  # No collaborative recommendations available
    
    # Fetch course names from content_data.csv based on collaborative recommendations
    recommended_course_names = content_data[content_data['course_id'].isin(collaborative_recommendations)]['course_nam'].tolist()
    
    return recommended_course_names

# Function to load user sector data
def load_user_sectors_data():
    user_sectors_data = pd.read_csv('first_type.csv')
    return user_sectors_data

# Function to get knowledge-based recommendations
def get_user_sector_recommendations(selected_user, user_sectors_data, content_data):
    user_sector = user_sectors_data[user_sectors_data['user_id'] == selected_user]['sectors'].tolist()

    if not user_sector:
        return []  # No sector information found for the selected user
    
    print(f"Selected User Sector: {user_sector[0]}")  # Debugging line
    
    # Use string matching to find courses similar to the user's sector
    content_data['similarity'] = content_data['course_nam'].apply(lambda x: fuzz.partial_ratio(x.lower(), user_sector[0].lower()))
    
    print(content_data[['course_nam', 'similarity']])  # Debugging line
    
    # Sort by similarity and get top recommendations
    recommended_courses = content_data.sort_values(by='similarity', ascending=False)['course_nam'].tolist()[:5]
    
    return recommended_courses

# Streamlit app code
st.markdown("#  Hybrid Coursera Course Recommendation System")
st.markdown("Find similar courses from a dataset")
st.markdown("Web App created by Sagar Bapodara")

# Load content-based data
content_data, tfidf_vectorizer, content_similarity = load_content_data()

# Load collaborative data
collaborative_data = load_collaborative_data()

# Load user sector data
user_sectors_data = load_user_sectors_data()

# Use a placeholder for course_list
course_list = []

if content_data is not None:
    column_name = 'course_nam'
    course_list = content_data[column_name].values if column_name in content_data.columns else []

selected_course = st.selectbox("Type or select a course you like:", course_list)

# Display selected course
st.write(f"Selected Course: {selected_course}")

# Option to choose between recommendation methods
recommendation_option = st.radio("Choose Recommendation Method:", ('Content-Based', 'Collaborative Filtering', 'Knowledge-Based'))

# Option to choose a user id for collaborative filtering
selected_user_id = st.number_input("Choose User ID for Collaborative Filtering:", min_value=1, max_value=collaborative_data['user_id'].max(), value=1)

# Display recommended courses based on the selected method
if st.button('Show Recommended Courses'):
    st.write("Recommended Courses based on your interests are:")

    if recommendation_option == 'Content-Based':
        recommended_course_names = content_based_recommend(selected_course, content_data, tfidf_vectorizer, content_similarity)
    elif recommendation_option == 'Collaborative Filtering':
        selected_user = selected_user_id
        collaborative_recommendations = get_user_collaborative_recommendations(selected_user, collaborative_data, content_data)
    
        if collaborative_recommendations:
            recommended_course_names = collaborative_recommendations
        else:
            recommended_course_names = []  # Handle the case when no collaborative recommendations are available
    else:  # Knowledge-Based
        selected_user = selected_user_id
        knowledge_based_recommendations = get_user_sector_recommendations(selected_user, user_sectors_data, content_data)
    
        if knowledge_based_recommendations:
            recommended_course_names = knowledge_based_recommendations
        else:
            recommended_course_names = []  # Handle the case when no knowledge-based recommendations are available

    if recommended_course_names:
        for i, course_name in enumerate(recommended_course_names, 1):
            st.text(f"{i}. {course_name}")
    else:
        st.text("No recommendations found.")

    st.text(" ")
   
