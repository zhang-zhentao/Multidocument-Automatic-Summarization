import os
import re
import numpy as np
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    text_block = re.findall(r'<TEXT>[\s\S]*?<\/TEXT>', content)[0]
    text_block = re.sub('<TEXT>\n', '', text_block)
    text_block = re.sub('\n<\/TEXT>', '', text_block)
    original_sentences = nltk.sent_tokenize(text_block.replace('\n', ' '))
    
    # Apply preprocessing here as required
    processed_sentences = original_sentences

    return original_sentences, processed_sentences

def generate_summary(sentence_similarity_matrix, original_sentences, processed_sentences, threshold, byte_limit):
    sentence_similarity_graph = {i: np.where(sentence_similarity_matrix[i] >= threshold)[0] for i in range(len(processed_sentences))}
    sorted_sentences = sorted(sentence_similarity_graph, key=lambda k: len(sentence_similarity_graph[k]), reverse=True)

    selected_sentences = []
    for index in sorted_sentences:
        candidate_sentence = original_sentences[index]
        is_similar = False

        for selected_sentence in selected_sentences:
            similarity = sentence_similarity_matrix[index][selected_sentences.index(selected_sentence)]
            if similarity >= threshold:
                is_similar = True
                break

        if not is_similar:
            selected_sentences.append(candidate_sentence)
            current_bytes = sum([len(sentence.encode('utf-8')) for sentence in selected_sentences])

            if current_bytes > byte_limit:
                selected_sentences.pop()
                break

    summary = ' '.join(selected_sentences)
    return summary

docs_path = '../../DUC04/unpreprocess data/docs/'
output_folder = '../Result/Sentence2VecSimilaritySummarizer'
threshold = 0.5
byte_limit = 665

model_path = os.path.abspath("../distilbert-base-nli-mean-tokens")
sentence_model = SentenceTransformer(model_path)

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

        sentence_embeddings = sentence_model.encode(processed_topic_sentences)
        sentence_similarity_matrix = cosine_similarity(sentence_embeddings)

        summary = generate_summary(sentence_similarity_matrix, original_topic_sentences, processed_topic_sentences, threshold, byte_limit)

        output_file = os.path.join(output_folder, f"{folder}-Sentence2VecSimilaritySummarizer.txt")
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
