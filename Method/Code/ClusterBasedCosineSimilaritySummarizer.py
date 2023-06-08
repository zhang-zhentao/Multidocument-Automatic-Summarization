import os
import re
import math
import nltk
import time
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    sentences = re.findall(r'<TEXT>[\s\S]*?<\/TEXT>', content)[0]
    sentences = re.sub('<TEXT>\n', '', sentences)
    sentences = re.sub('\n<\/TEXT>', '', sentences)
    sentences = sentences.replace('\n', ' ')
    original_sentences = nltk.sent_tokenize(sentences)
    stopwords_list = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    processed_sentences = []
    for sentence in original_sentences:
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stopwords_list]
        words = [stemmer.stem(word) for word in words]
        processed_sentence = ' '.join(words)
        processed_sentences.append(processed_sentence)
    return original_sentences, processed_sentences

def generate_improved_summary(original_sentences, processed_sentences, vectorizer, byte_limit):
    X = vectorizer.fit_transform(processed_sentences)
    n_clusters = int(math.sqrt(len(processed_sentences)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    cluster_sentence_similarities = {}
    cluster_center_similarities = []
    for i in range(n_clusters):
        cluster_original_sentences = [original_sentences[j] for j in range(len(processed_sentences)) if cluster_labels[j] == i]
        cluster_processed_sentences = [processed_sentences[j] for j in range(len(processed_sentences)) if cluster_labels[j] == i]
        cluster_vectors = vectorizer.transform(cluster_processed_sentences)
        cluster_similarities = cosine_similarity(cluster_vectors, cluster_centers[i].reshape(1, -1)).flatten()
        cluster_sentence_similarities[i] = cluster_similarities
        cluster_center_similarity = np.mean(cluster_similarities)
        cluster_center_similarities.append((i, cluster_center_similarity))

    cluster_center_similarities.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = []
    current_bytes = 0
    max_rounds = len(processed_sentences) // n_clusters + 1
    for _ in range(max_rounds):
        for cluster_index, _ in cluster_center_similarities:
            cluster_sentences = [(j, original_sentences[j]) for j in range(len(processed_sentences)) if cluster_labels[j] == cluster_index]
            cluster_sentence_similarities[cluster_index] = sorted(cluster_sentence_similarities[cluster_index], reverse=True)
            while len(cluster_sentence_similarities[cluster_index]) > 0:
                highest_similarity_index = np.argmax(cluster_sentence_similarities[cluster_index])
                selected_sentence = cluster_sentences[highest_similarity_index][1]
                selected_sentence_bytes = len(selected_sentence.encode('utf-8'))
                if current_bytes + selected_sentence_bytes <= byte_limit:
                    selected_sentences.append(selected_sentence)
                    current_bytes += selected_sentence_bytes
                    break
                else:
                    cluster_sentence_similarities[cluster_index] = np.delete(cluster_sentence_similarities[cluster_index], highest_similarity_index)
                    cluster_sentences.pop(highest_similarity_index)
            if current_bytes >= byte_limit:
                break
        if current_bytes >= byte_limit:
            break

    summary = ' '.join(selected_sentences)
    return summary

docs_path = '../../DUC04/unpreprocess data/docs/'
output_folder = '../Result/ClusterBasedCosineSimilaritySummarizer'
byte_limit = 665

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

topic_count = 0
start_time = time.time()

for folder in os.listdir(docs_path):
    folder_path = os.path.join(docs_path, folder)
    if os.path.isdir(folder_path):
        original_topic_sentences = []
        processed_topic_sentences = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            original_sentences, processed_sentences = preprocess(file_path)
            original_topic_sentences.extend(original_sentences)
            processed_topic_sentences.extend(processed_sentences)

        vectorizer = TfidfVectorizer()
        summary = generate_improved_summary(original_topic_sentences, processed_topic_sentences, vectorizer, byte_limit)

        output_file = os.path.join(output_folder, f"{folder}-ClusterBasedCosineSimilaritySummarizer.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Summary for topic {folder}:\n")
            f.write(summary)

        topic_count += 1
        if topic_count % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {topic_count} topics in {elapsed_time:.2f} seconds...")

total_elapsed_time = time.time() - start_time
print(f"Total elapsed time: {total_elapsed_time:.2f} seconds.")
print("All topics processed successfully.")
