import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from deviation_loss import DeviationLoss
from binary_focal_loss import BinaryFocalLoss
from tqdm import tqdm, trange
from balanced_sampler import BalancedBatchSampler, worker_init_fn_seed
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from generate_fewshot_outliers import GenerateFewShotOutliers
from utils import get_sentences, CustomDataset

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Few-Shot Anomaly Detection")
parser.add_argument('--learning_rate', type=float, default=1e-6)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=16)
parser.add_argument('--evaluate_during_training_steps', type=int, default=600)
parser.add_argument('--contamination', type=int, default=0)
parser.add_argument('--top_k', type=float, default=0.1)
parser.add_argument('--few_shot_anomalies', type=int, default=10)
parser.add_argument('--include_regularization', type=int, default=1)
parser.add_argument('--dataset', type=str, default="ag")
parser.add_argument('--inlier_class', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--attention_size', type=int, default=150)
parser.add_argument('--num_heads', type=int, default=5)


args = parser.parse_args()
learning_rate = args.learning_rate
max_seq_len = args.max_seq_len
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
evaluate_during_training_steps = args.evaluate_during_training_steps
contamination = args.contamination
top_k = args.top_k
few_shot_anomalies = args.few_shot_anomalies
include_regularization = args.include_regularization
dataset = args.dataset
inlier_class = args.inlier_class
num_workers = args.num_workers
attention_size = args.attention_size
num_heads = args.num_heads

class SBERTWithAttention(nn.Module):
    def __init__(self, config):
        super(SBERTWithAttention, self).__init__()
        self.sbert = SentenceTransformer(config)

        self.hidden_size = self.sbert.get_sentence_embedding_dimension()

        # hyperparameters
        self.attention_size = attention_size
        self.num_heads = num_heads

        self.W1 = nn.Linear(self.hidden_size, self.attention_size, bias=False)    
        self.W2 = nn.Linear(self.attention_size, self.num_heads, bias=False)

    def forward(self, features):
        sbert_output = self.sbert(features)
        
        hidden_state = sbert_output['token_embeddings']

        t = torch.tanh(self.W1(hidden_state))
        t = F.softmax(self.W2(t), dim=1)
        attention_matrix = t.transpose(1, 2)

        outputs = attention_matrix @ hidden_state
        outputs = torch.flatten(outputs, start_dim=1)
        
        # selecting top k% scores and taking mean
        k = 0.1
        topk = max(int(outputs.size(1) * k), 1)
        outputs = torch.topk(torch.abs(outputs), topk, dim=1)[0]

        outputs = outputs.mean(dim=1)
        outputs = outputs.float()

        return outputs, attention_matrix

class FATEModel():
    def __init__(self) -> None:
        self.subsets= {
            "ag": ["business", "sci", "sports", "world"],
            "20ng": ["comp", "rec", "sci", "misc", "pol", "rel"],
            "reuters": ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship"]
        }
        self.datasets = list(self.subsets.keys())

        self.subset = self.subsets[dataset][inlier_class]

        # making training outliers file
        gfs = GenerateFewShotOutliers(dataset, self.subset)

        inlier_file_path = f"../datasets/{dataset}/train/{self.subset}.txt"

        # 0, 5, 10 or 15% contamination
        if contamination:
            print(f"Contamination: {contamination}%")
            inlier_file_path = f"../datasets/{dataset}/train/{self.subset}-contaminated/{self.subset}_c{contamination}.txt"

        # few-shot anomalies
        anom_file_path = f"../datasets/{dataset}/train/{self.subset}-outliers.txt"


        test_inlier_file_path = f"../datasets/{dataset}/test/{self.subset}.txt"
        test_anom_file_path = f"../datasets/{dataset}/test/{self.subset}-outliers.txt"

        # training sentences
        self.inlier_sentences = get_sentences(inlier_file_path)
        self.anom_sentences = get_sentences(anom_file_path)

        # testing sentences
        self.test_inlier_sentences = get_sentences(test_inlier_file_path)
        self.test_anom_sentences = get_sentences(test_anom_file_path)


        print("\nInlier sentences: ", len(self.inlier_sentences))
        print("Anomalous sentences: ", len(self.anom_sentences))
        print("Test inlier sentences: ", len(self.test_inlier_sentences))
        print("Test anomalous sentences: ", len(self.test_anom_sentences))

        # choose loss function here
        self.criterion = DeviationLoss()
        # criterion = BinaryFocalLoss()
        # criterion = nn.BCEWithLogitsLoss()

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SBERTWithAttention("all-MiniLM-L6-v2")
        self.model = self.model.to(self.device)

        # fetching dataloaders
        self.train_dataloader, self.test_inlier_dataloader, self.test_anom_dataloader = self.get_dataloaders()

        self.num_epochs = 4 if dataset == "ag" else 50
        if dataset == "reuters": self.num_epochs = 40
        self.evaluate_during_training_steps = evaluate_during_training_steps
        

        # helps in case of datasets like reuters, where less sentences are available per subset as train inliers
        if len(self.inlier_sentences) < 500: 
            self.num_epochs = 80
            self.evaluate_during_training_steps = 100


    def get_attention_masks(self, sentences):

        # encode the sentences using the tokenizer
        tokenized_output = self.tokenizer(sentences, max_length = 128, padding='max_length', truncation= True)
        input_ids = tokenized_output.input_ids
        attention_masks = tokenized_output.attention_mask
            
        return input_ids, attention_masks

    def get_dataloaders(self):
        input_ids_inlier, attention_masks_inlier = self.get_attention_masks(self.inlier_sentences)
        input_ids_anom, attention_masks_anom = self.get_attention_masks(self.anom_sentences)

        input_ids = input_ids_inlier + input_ids_anom
        attention_masks = attention_masks_inlier + attention_masks_anom
        labels = np.array(np.zeros(len(input_ids_inlier)).tolist() + np.ones(len(input_ids_anom)).tolist())


        normal_idx = np.argwhere(labels == 0).flatten()

        outlier_idx = np.argwhere(labels == 1).flatten()

        # convert input_sentences, attention_masks, and labels to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels)

        # create a TensorDataset from input_ids, attention_masks, and labels
        train_dataset = CustomDataset(input_ids, attention_masks, labels, normal_idx, outlier_idx)

        # custom sampler to have equal number of inliers and anomalies in each batch
        train_sampler = BalancedBatchSampler(train_dataset, train_batch_size)

        # initialize the dataloader with batch_size = 16
        train_dataloader = DataLoader(train_dataset, 
                                    worker_init_fn=worker_init_fn_seed,
                                    batch_sampler=train_sampler,
                                    num_workers=num_workers
                                    )

        # getting testing dataset - inlier
        test_input_ids_inlier, test_attention_masks_inlier = self.get_attention_masks(self.test_inlier_sentences)
        test_labels_inlier = np.array(np.zeros(len(test_input_ids_inlier)).tolist())

        test_input_ids_inlier = torch.tensor(test_input_ids_inlier)
        test_attention_masks_inlier = torch.tensor(test_attention_masks_inlier)
        test_labels_inlier = torch.tensor(test_labels_inlier)

        test_inlier_dataset = CustomDataset(test_input_ids_inlier, test_attention_masks_inlier, test_labels_inlier)
        test_inlier_sampler = SequentialSampler(test_inlier_dataset)
        test_inlier_dataloader = DataLoader(test_inlier_dataset,
                                    sampler=test_inlier_sampler,
                                    batch_size=test_batch_size,
                                    num_workers=num_workers)

        # getting testing dataset - outlier
        test_input_ids_anom, test_attention_masks_anom = self.get_attention_masks(self.test_anom_sentences)
        test_labels_anom = np.array(np.ones(len(test_input_ids_anom)).tolist())

        test_input_ids_anom = torch.tensor(test_input_ids_anom)
        test_attention_masks_anom = torch.tensor(test_attention_masks_anom)
        test_labels_anom = torch.tensor(test_labels_anom)

        test_anom_dataset = CustomDataset(test_input_ids_anom, test_attention_masks_anom, test_labels_anom)
        test_anom_sampler = SequentialSampler(test_anom_dataset)
        test_anom_dataloader = DataLoader(test_anom_dataset,
                                    sampler=test_anom_sampler,
                                    batch_size=test_batch_size,
                                    num_workers=num_workers)
        
        return train_dataloader, test_inlier_dataloader, test_anom_dataloader
        
    # function for running test_model() and getting scores
    def calculate_scores(self, dataloader):
        
        dev_scores = []

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                labels = labels.float()

                features = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                outputs, A = self.model(features)

                identity_mat = torch.eye(5).to(self.device)
                CCT = A @ A.transpose(1, 2)
                loss_2 = torch.mean((CCT.squeeze() - identity_mat) ** 2)

                if include_regularization:
                    loss = self.criterion(outputs, labels) + loss_2
                
                else: loss = self.criterion(outputs, labels)

                print(f"\rtesting loss: {loss:.4f}", end="")

                scores = outputs.cpu().numpy()

                if len(dev_scores) == 0:
                    dev_scores = scores

                else:
                    dev_scores = np.concatenate((dev_scores, scores), axis=0)
        
        return dev_scores

    def get_metrics(self, gt, preds, invert=False):
        fpr, tpr, roc_thresholds = roc_curve(gt, preds)
        cutoff = np.argmax(tpr-fpr)

        roc_auc = auc(fpr, tpr)

        if invert:
            precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(gt, -preds, pos_label=0)
        else:
            precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(gt, preds)

        pr_auc_norm = auc(recall_norm, precision_norm)

        if invert:
            precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(gt, preds)
        else:
            precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(gt, -preds, pos_label=0)

        pr_auc_anom = auc(recall_anom, precision_anom)

        return roc_auc, pr_auc_norm, pr_auc_anom, roc_thresholds[cutoff]


    def get_preds_calc_metrics(self,
                            normal_scores,
                            anomaly_scores,
                            name='DevLoss',
                            invert=False):

        labels_normal = np.zeros_like(normal_scores)
        labels_anomaly = np.ones_like(anomaly_scores)

        gt = np.concatenate((labels_normal, labels_anomaly))
        preds = np.concatenate((normal_scores, anomaly_scores))

        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)

        roc_auc, pr_auc_norm, pr_auc_anom, cutoff = self.get_metrics(gt, preds, invert)
        real_preds = (roc_auc, pr_auc_norm, pr_auc_anom)

        # print(name)
        print(f'ROC-AUC    : {roc_auc:4f}')
        print(f'PR-AUC-in  : {pr_auc_norm:4f}')
        print(f'PR-AUC-out : {pr_auc_anom:4f}')

        return real_preds, cutoff

    # testing function
    def test_model(self):

        results = {}

        self.model.eval()
        
        # finding anomaly scores
        anom_scores = self.calculate_scores(self.test_anom_dataloader)

        # finding inlier scores
        inlier_scores = self.calculate_scores(self.test_inlier_dataloader)

        real_preds, cutoff = self.get_preds_calc_metrics(
                        inlier_scores,
                        anom_scores,
                        name='NE',
                        invert=True)

        results["DEV/roc_auc"], results["DEV/pr_auc_in"], results["DEV/pr_auc_out"] = real_preds

    def train_model(self):

        train_iterator = trange(int(self.num_epochs),
                        desc="Epoch",
                        mininterval=0)
        global_step = 0

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        for epoch in train_iterator:
            for i, (input_ids, attention_mask, labels) in enumerate(tqdm(self.train_dataloader, desc=f"iteration{global_step}")):
                
                optimizer.zero_grad()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                labels = labels.to(self.device)
                labels = labels.float()

                features = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                outputs, A = self.model(features)

                identity_mat = torch.eye(5).to(self.device)
                CCT = A @ A.transpose(1, 2)
                loss_2 = torch.mean((CCT.squeeze() - identity_mat) ** 2)

                if include_regularization:
                    loss = self.criterion(outputs, labels) + loss_2
                    
                else: loss = self.criterion(outputs, labels)

                print(f"\rloss: {loss:.4f}", end="")

                loss.backward()
                optimizer.step()
                global_step+=1

                if global_step%self.evaluate_during_training_steps == 0: self.test_model()

def main():
    fm = FATEModel()

    # test model before training
    fm.test_model()

    # train the model
    fm.train_model()

    # test final model
    fm.test_model()

    print("Training and testing completed.")

if __name__ == '__main__':
    main()