import os
import re
import time
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def extract_first_sentence(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    sentences = re.findall(r'<TEXT>[\s\S]*?<\/TEXT>', content)[0]
    sentences = re.sub('<TEXT>\n', '', sentences)
    sentences = re.sub('\n<\/TEXT>', '', sentences)
    sentences = sentences.replace('\n', ' ')
    sentences = sent_tokenize(sentences)
    return sentences[0]

docs_path = '../../DUC04/unpreprocess data/docs/'
output_folder = '../Result/Baseline'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

topic_count = 0
start_time = time.time()

for folder in os.listdir(docs_path):
    folder_path = os.path.join(docs_path, folder)
    if os.path.isdir(folder_path):
        baseline_summary_sentences = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            first_sentence = extract_first_sentence(file_path)
            baseline_summary_sentences.append(first_sentence)

        baseline_summary = ' '.join(baseline_summary_sentences)
        output_file = os.path.join(output_folder, f"{folder}-Baseline.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Baseline summary for topic {folder}:\n")
            f.write(baseline_summary)

        topic_count += 1
        if topic_count % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {topic_count} topics in {elapsed_time:.2f} seconds...")

total_elapsed_time = time.time() - start_time
print(f"Total elapsed time: {total_elapsed_time:.2f} seconds.")
print("All topics processed successfully.")
