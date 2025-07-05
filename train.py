import torch
import wandb
import pickle
import importlib.util
import argparse
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from main.arguments.utils import eval
from main.preprocessor.utils.modify import prepare_dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=r"src\arguments\train_args.py", help="Path to config file (Python dict)")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("train_args", args.config)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
args_config = config_module.args_config
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
torch.manual_seed(42)

wandb.login(key="48a4efb01e5dfdf3c64fab49e7d839cea68fd508")

with open('./data/processed/queries_after.pkl', 'rb') as f:
    queries = pickle.load(f)
with open('./data/processed/corpus_after.pkl', 'rb') as f:
    corpus = pickle.load(f)
with open('./data/processed/relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)

pairs = prepare_dataset(queries, corpus, relevant_docs)
evaluator = eval(queries, corpus, relevant_docs)

model_id = "NghiemAbe/Vi-Legal-Bi-Encoder-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = SentenceTransformer(model_id)

inner_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(model, inner_loss, matryoshka_dims=[512, 256, 128, 64])

wandb.init(
    project="sentence_xnk",
    name=args_config.get("run_name", "experiment_default"),
    config=args_config
)

train_args = SentenceTransformerTrainingArguments(**args_config)
trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=pairs,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
trainer.save_model(r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval\model")
