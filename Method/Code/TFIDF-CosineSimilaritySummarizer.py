import os
import re
import nltk
import time
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.chdir(os.path.dirname(os.path.abspath(__file__)))
docs_path = '../../DUC04/unpreprocess data/docs/'
threshold = 0.5
byte_limit = 665
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



def generate_summary(sentence_similarity_graph, original_sentences, processed_sentences, vectorizer, threshold, byte_limit):
    selected_sentences = []
    sorted_sentences = sorted(sentence_similarity_graph, key=lambda k: len(sentence_similarity_graph[k]), reverse=True)

    for index in sorted_sentences:
        candidate_sentence = processed_sentences[index]
        candidate_vector = vectorizer.transform([candidate_sentence])
        is_similar = False

        for selected_sentence in selected_sentences:
            selected_index = original_sentences.index(selected_sentence)
            selected_vector = vectorizer.transform([processed_sentences[selected_index]])
            similarity = cosine_similarity(candidate_vector, selected_vector)
            if similarity >= threshold:
                is_similar = True
                break

        if not is_similar:
            selected_sentences.append(original_sentences[index])
            current_bytes = sum([len(sentence.encode('utf-8')) for sentence in selected_sentences])

            if current_bytes > byte_limit:
                selected_sentences.pop()
                break

    summary = ' '.join(selected_sentences)
    return summary

output_folder = '../Result/TFIDF-CosineSimilaritySummarizer'
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
        X = vectorizer.fit_transform(processed_topic_sentences)
        sentence_similarity_matrix = cosine_similarity(X)
        sentence_similarity_graph = {i: np.where(sentence_similarity_matrix[i] >= threshold)[0] for i in range(len(processed_topic_sentences))}

        summary = generate_summary(sentence_similarity_graph, original_topic_sentences, processed_topic_sentences, vectorizer, threshold, byte_limit)

        output_file = os.path.join(output_folder, f"{folder}-TFIDF-CosineSimilaritySummarizer.txt")
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