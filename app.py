import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

dataset_url = "https://drive.google.com/uc?export=download&id=1Cd8UZKDfcZxi8c1TwoNkvT7jRmbrS5on"
dataset_local_path = ".\online_course_recommendation.csv"

def baixar_csv_se_necessario(local_path, url):
    if not os.path.exists(local_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)

def carregar_dados():
    baixar_csv_se_necessario(dataset_local_path, dataset_url)
    df = pd.read_csv(dataset_local_path)
    return df

def processar_texto(courses):
    courses['text'] = courses[['description', 'category', 'tags']].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(courses['text'])
    return tfidf_matrix, tfidf

def construir_matriz_usuario_item(df):
    df['rating'] = 5
    return df.pivot_table(index='user_id', columns='course_id', values='rating', fill_value=0)

def reduzir_dimensionalidade(matriz):
    svd = TruncatedSVD(n_components=50, random_state=42)
    return svd.fit_transform(matriz), svd

def agrupar_usuarios(user_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_features)
    return clusters, kmeans

def cursos_mais_populares(cluster_id, user_cluster_df, df, top_n=5):
    usuarios_do_cluster = user_cluster_df[user_cluster_df['cluster'] == cluster_id]['user_id']
    cluster_data = df[df['user_id'].isin(usuarios_do_cluster)]
    top_courses = cluster_data['course_id'].value_counts().head(top_n).index.tolist()
    return top_courses

def recomendar_hibrido(user_id, user_course_matrix, df, tfidf_matrix, user_cluster_df, courses):
    if user_id in user_course_matrix.index:
        user_rated = user_course_matrix.loc[user_id]
        liked_courses = user_rated[user_rated >= 4].index.tolist()

        if not liked_courses:
            cluster_id = user_cluster_df[user_cluster_df['user_id'] == user_id]['cluster'].values[0]
            return cursos_mais_populares(cluster_id, user_cluster_df, df)

        liked_indices = [courses[courses['course_id'] == cid].index[0] for cid in liked_courses if cid in courses['course_id'].values]
        if not liked_indices:
            return []

        mean_vector = tfidf_matrix[liked_indices].mean(axis=0)
        mean_vector = np.asarray(mean_vector)
        sim_scores = cosine_similarity(mean_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(sim_scores)[::-1]
        recomendados = courses.iloc[top_indices]['course_id']
        return recomendados[~recomendados.isin(liked_courses)].head(5).tolist()
    else:
        cluster_id = 0
        return cursos_mais_populares(cluster_id, user_cluster_df, df)

st.title("Recomendador de Cursos Online")

with st.spinner("Carregando dados..."):
    df = carregar_dados()
    tfidf_matrix, tfidf = processar_texto(df)
    user_course_matrix = construir_matriz_usuario_item(df)
    user_features, svd = reduzir_dimensionalidade(user_course_matrix)
    user_clusters, kmeans = agrupar_usuarios(user_features)
    user_cluster_df = pd.DataFrame({'user_id': user_course_matrix.index, 'cluster': user_clusters})

user_ids = user_course_matrix.index.tolist()
user_id = st.selectbox("Selecione um ID de usuário para recomendar:", user_ids + ['Usuário Novo'])

if st.button("Recomendar Cursos"):
    if user_id == 'Usuário Novo':
        user_input = 'novo'
    else:
        user_input = user_id

    recomendacoes = recomendar_hibrido(user_input, user_course_matrix, df, tfidf_matrix, user_cluster_df, df)
    
    textos = []
    for cid in recomendacoes:
      curso = df[df['course_id'] == cid].iloc[0]
      textos.append(str(curso['description']) + " " + str(curso['category']) + " " + str(curso['tags']))
    texto_total = " ".join(textos)

    if texto_total.strip():
      st.subheader("Word Cloud dos Parâmetros de Recomendação")
      wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_total)
      fig, ax = plt.subplots(figsize=(10, 5))
      ax.imshow(wordcloud, interpolation='bilinear')
      ax.axis('off')
      st.pyplot(fig)

    if recomendacoes:
        st.subheader("Cursos Recomendados:")
        for cid in recomendacoes:
            curso = df[df['course_id'] == cid].iloc[0]
            st.markdown(f"**{curso['course_id']}**")
            st.markdown(f"_Categoria:_ {curso['category']}")
            st.markdown(f"_Tags:_ {curso['tags']}")
            st.markdown(f"{curso['description'][:200]}...")
            st.markdown("---")
    else:
        st.warning("Nenhuma recomendação encontrada para este usuário.")

