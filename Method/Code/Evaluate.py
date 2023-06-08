import os
import re
from rouge import Rouge

method_list=["Baseline","ClusterBasedCosineSimilaritySummarizer","Sentence2VecSimilaritySummarizer","TFIDF-CosineSimilaritySummarizer"]
method = method_list[0]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content

def calculate_rouge_scores(hypothesis, references):
    rouge = Rouge()
    hypothesis = " ".join(hypothesis.split())  # Remove extra whitespaces

    scores_list = []
    for ref in references:
        ref = " ".join(ref.split())  # Remove extra whitespaces
        scores = rouge.get_scores(hypothesis, ref)
        scores_list.append(scores[0])

    if not scores_list:
        return {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    avg_scores = {key: {metric: 0 for metric in scores_list[0][key]} for key in scores_list[0]}
    for score_dict in scores_list:
        for key in score_dict:
            for metric in score_dict[key]:
                avg_scores[key][metric] += score_dict[key][metric]

    for key in avg_scores:
        for metric in avg_scores[key]:
            avg_scores[key][metric] /= len(scores_list)

    return avg_scores

model_path = os.path.join("..", "..", "DUC04", "model", "04model")
generated_summaries_path = os.path.join("..", "Result", method)
total_rouge_1_r = 0
total_rouge_1_p = 0
total_rouge_1_f = 0
total_rouge_2_r = 0
total_rouge_2_p = 0
total_rouge_2_f = 0
total_rouge_l_r = 0
total_rouge_l_p = 0
total_rouge_l_f = 0
total_summaries = 0


generated_summary_files = os.listdir(generated_summaries_path)
generated_summary_pattern = re.compile(r"d3(\d+)t-.*\.txt")

for gen_summary_file in generated_summary_files:
    match = generated_summary_pattern.match(gen_summary_file)
    if not match:
        continue

    topic_num = int(match.group(1))
    generated_summary_file = os.path.join(generated_summaries_path, gen_summary_file)
    generated_summary = read_file(generated_summary_file)

    reference_summaries = []
    reference_files = os.listdir(model_path)
    reference_pattern = re.compile(f"D3{topic_num:04}.M.100.T.\\w")
    matched_ref_files = []
    for ref_file in reference_files:
        if reference_pattern.match(ref_file):
            reference_summary = read_file(os.path.join(model_path, ref_file))
            reference_summaries.append(reference_summary)
            matched_ref_files.append(ref_file)

    rouge_scores = calculate_rouge_scores(generated_summary, reference_summaries)

    #打印文件名
    # print(f"Generated summary file: {generated_summary_file}")
    # print("Reference summary files:")
    # for ref_file in matched_ref_files:
    #     print(f"  - {ref_file}")

    total_rouge_1_r += rouge_scores['rouge-1']['r']
    total_rouge_1_p += rouge_scores['rouge-1']['p']
    total_rouge_1_f += rouge_scores['rouge-1']['f']
    total_rouge_2_r += rouge_scores['rouge-2']['r']
    total_rouge_2_p += rouge_scores['rouge-2']['p']
    total_rouge_2_f += rouge_scores['rouge-2']['f']
    total_rouge_l_r += rouge_scores['rouge-l']['r']
    total_rouge_l_p += rouge_scores['rouge-l']['p']
    total_rouge_l_f += rouge_scores['rouge-l']['f']

    total_summaries += 1

average_rouge_1_r = total_rouge_1_r / total_summaries
average_rouge_1_p = total_rouge_1_p / total_summaries
average_rouge_1_f = total_rouge_1_f / total_summaries
average_rouge_2_r = total_rouge_2_r / total_summaries
average_rouge_2_p = total_rouge_2_p / total_summaries
average_rouge_2_f = total_rouge_2_f / total_summaries
average_rouge_l_r = total_rouge_l_r / total_summaries
average_rouge_l_p = total_rouge_l_p / total_summaries
average_rouge_l_f = total_rouge_l_f / total_summaries

print(f"For {method}:")
print("Average Rouge-1 Recall: ", average_rouge_1_r)
print("Average Rouge-1 Precision: ", average_rouge_1_p)
print("Average Rouge-1 F-score: ", average_rouge_1_f)
print("Average Rouge-2 Recall: ", average_rouge_2_r)
print("Average Rouge-2 Precision: ", average_rouge_2_p)
print("Average Rouge-2 F-score: ", average_rouge_2_f)
print("Average Rouge-L Recall: ", average_rouge_l_r)
print("Average Rouge-L Precision: ", average_rouge_l_p)
print("Average Rouge-L F-score: ", average_rouge_l_f)

rouge_1_weight = 1/3
rouge_2_weight = 1/3
rouge_l_weight = 1/3

# 使用F1分数计算加权平均
overlap_score = (
    rouge_1_weight * average_rouge_1_f +
    rouge_2_weight * average_rouge_2_f +
    rouge_l_weight * average_rouge_l_f
)

print("Overlap score: ", overlap_score)
