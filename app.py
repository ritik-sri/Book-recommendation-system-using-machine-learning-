import pickle
import streamlit as st
import numpy as np
st.header("Book Recommender System using Machine Learning")


model = pickle.load(open('models/model.pkl', 'rb'))
book_names = pickle.load(open('models/book_names.pkl', 'rb'))
final_rating = pickle.load(open('models/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('models/book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):

    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url


def recommend_books(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestion)

    n = len(suggestion)
    for i in range(n):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)

    return book_list, poster_url


selected_books = st.selectbox("Type or Select a book",
                              book_names)

if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendation_books[0])   # [0] will be the same book as in the search box
        st.image(poster_url[0])

    with col2:
        st.text(recommendation_books[1])   
        st.image(poster_url[1])

    with col3:
        st.text(recommendation_books[2])   
        st.image(poster_url[2])

    with col4:
        st.text(recommendation_books[3])   
        st.image(poster_url[3])

    with col5:
        st.text(recommendation_books[4])   
        st.image(poster_url[4])
