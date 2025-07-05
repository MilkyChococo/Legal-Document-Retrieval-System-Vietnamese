from tqdm import tqdm
from sentence_transformers.util import cos_sim as consine
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from datasets import Dataset

def eval(queries,corpus,relevant_docs):
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    matryoshka_evaluators = []

    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": consine},
        )
        matryoshka_evaluators.append(ir_evaluator)
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    return evaluator
