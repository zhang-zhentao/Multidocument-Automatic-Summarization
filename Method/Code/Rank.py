import os
import re
import numpy as np
from collections import defaultdict
from rouge import Rouge
import pandas as pd
import openpyxl

script_folder = os.path.dirname(os.path.abspath(__file__))

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

def calculate_overlap_score(avg_rouge_1_f, avg_rouge_2_f, avg_rouge_l_f):
    rouge_1_weight = 1/3
    rouge_2_weight = 1/3
    rouge_l_weight = 1/3

    overlap_score = (
        rouge_1_weight * avg_rouge_1_f +
        rouge_2_weight * avg_rouge_2_f +
        rouge_l_weight * avg_rouge_l_f
    )

    return overlap_score

# Read files
input_folder = os.path.join(script_folder, "..", "..", "DUC04", "participating systems result", "2")
summaries = defaultdict(lambda: defaultdict(list))

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    summary_content = read_file(file_path)

    topic_id, team_id = re.match(r"D(\d+)\.M\.100\.T\.([A-Z\d]+)", file_name).groups()
    summaries[topic_id][team_id].append(summary_content)

# Calculate scores
team_scores = defaultdict(list)

for topic_id, summaries_by_team in summaries.items():
    reference_summaries = [summary for team_id, summary_list in summaries_by_team.items() if not team_id.isdigit() for summary in summary_list]

    for team_id, team_summaries in summaries_by_team.items():
        if team_id.isdigit():
            topic_scores = []

            for summary in team_summaries:
                rouge_scores = calculate_rouge_scores(summary, reference_summaries)
                overlap_score = calculate_overlap_score(
                    rouge_scores['rouge-1']['f'],
                    rouge_scores['rouge-2']['f'],
                    rouge_scores['rouge-l']['f']
                )
                topic_scores.append(overlap_score)

            team_scores[team_id].append(np.mean(topic_scores))

# Calculate scores
team_scores = defaultdict(list)

for topic_id, summaries_by_team in summaries.items():
    reference_summaries = [summary for team_id, summary_list in summaries_by_team.items() if not team_id.isdigit() for summary in summary_list]

    for team_id, team_summaries in summaries_by_team.items():
        if team_id.isdigit():
            topic_scores = []

            for summary in team_summaries:
                rouge_scores = calculate_rouge_scores(summary, reference_summaries)
                overlap_score = calculate_overlap_score(
                    rouge_scores['rouge-1']['f'],
                    rouge_scores['rouge-2']['f'],
                    rouge_scores['rouge-l']['f']
                )
                topic_scores.append(overlap_score)

            team_scores[team_id].append(np.mean(topic_scores))

# Calculate average scores for each team
average_team_scores = {team_id: np.mean(scores) for team_id, scores in team_scores.items()}

# Print results
for team_id, avg_score in sorted(average_team_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"Team {team_id}: Average Overlap Score: {avg_score:.4f}")

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Team ID", "Average Overlap Score"])

# Add the results to the DataFrame
for team_id, avg_score in sorted(average_team_scores.items(), key=lambda x: x[1], reverse=True):
    results_df = results_df.append({"Team ID": team_id, "Average Overlap Score": avg_score}, ignore_index=True)

# Save the DataFrame to an Excel file
results_df.to_excel(os.path.join(script_folder,"..","Result", "results.xlsx"), index=False)

