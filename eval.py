from main.evaluation.utils import compute_metrics_with_score
import pandas as pd
import os
ranking_file = 'main/evaluation/ranking_10_base_model.csv'
test_file = r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data4pr\test.csv'
if os.path.exists(ranking_file) and os.path.exists(test_file):
    ranking_df = pd.read_csv(ranking_file)
    test_df = pd.read_csv(test_file)
    predictions = {}
    for _, row in ranking_df.iterrows():
        qid = row['qid']
        pid = row['cid']
        score = row['Score']
        if qid not in predictions:
            predictions[qid] = []
        predictions[qid].append((pid, score))

    ground_truth = {str(row['qid']): str(row['cid']) for _, row in test_df.iterrows()}
    metrics = compute_metrics_with_score(predictions, ground_truth, k=10)
    print(metrics)