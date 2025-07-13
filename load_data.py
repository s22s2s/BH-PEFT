import random
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, precision_recall_fscore_support, \
    accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import RobertaTokenizer, PreTrainedTokenizerBase
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from dataclasses import dataclass
class load_drug_reviews():
    def __init__(self, data_name,tokenizer=None):
        self.train_data_df = pd.read_csv(data_name+'/drugsComTrain_raw.tsv',sep='\t')
        self.eval_data_df = pd.read_csv(data_name + '/drugsComTest_raw.tsv', sep='\t')
        self.tokenizer = tokenizer

    def preprocess(self,examples):
        tokenized = self.tokenizer(examples['content'], truncation=True, padding=True)
        tokenized['labels'] = examples['normalized_score']  # 使用归一化的得分作为标签
        # del examples['normalized_score']
        return tokenized

    def load_data(self,test=False):
        scaler = StandardScaler()

        self.train_data_df = self.train_data_df.drop(columns=['Unnamed: 0', 'drugName','condition','date','usefulCount'])
        self.eval_data_df = self.eval_data_df.drop(columns=['Unnamed: 0', 'drugName','condition','date','usefulCount'])

        self.train_data_df.rename(columns={'rating': 'normalized_score'}, inplace=True)
        # df['normalized_score'] = df['rating']
        self.train_data_df.rename(columns={'review': 'content'}, inplace=True)
        # df['content'] = df['body']
        self.eval_data_df.rename(columns={'rating': 'normalized_score'}, inplace=True)
        # df['normalized_score'] = df['rating']
        self.eval_data_df.rename(columns={'review': 'content'}, inplace=True)


        train_scores = self.train_data_df[['normalized_score']]
        scaler.fit(train_scores)

        self.train_data_df['normalized_score'] = scaler.transform(self.train_data_df[['normalized_score']])
        self.eval_data_df['normalized_score'] = scaler.transform(self.eval_data_df[['normalized_score']])
        val_df, test_df = train_test_split(self.eval_data_df, test_size=0.5, random_state=42)

        train_dataset = Dataset.from_pandas(self.train_data_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        if not test:
            train_dataset = train_dataset.map(self.preprocess, batched=True,
                                              remove_columns=["content", "normalized_score"])
            val_dataset = val_dataset.map(self.preprocess, batched=True,
                                          remove_columns=["content", "normalized_score", '__index_level_0__'])
            test_dataset = test_dataset.map(self.preprocess, batched=True,
                                            remove_columns=["content", "normalized_score", '__index_level_0__'])
        else:
            test_dataset = test_dataset.map(self.preprocess, batched=True,
                                            remove_columns=["content", "normalized_score", '__index_level_0__'])
            return test_dataset

        return train_dataset, val_dataset, test_dataset

class load_Ag_news():
    def __init__(self, data_name,tokenizer=None):
        self.dataset = load_dataset(data_name)
        self.tokenizer = tokenizer

    def preprocess(self,examples):
        tokenized = self.tokenizer(examples['text'], truncation=True, padding=True)
        return tokenized

    def load_data(self,test=False):
        tokenized_dataset = self.dataset.map(self.preprocess, batched=True, remove_columns=["text"])
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['test'].shard(num_shards=2, index=0)
        test_dataset = tokenized_dataset['test'].shard(num_shards=2, index=1)

        return train_dataset, eval_dataset, test_dataset

    def id2label(self):
        num_labels = self.dataset['train'].features['label'].num_classes
        class_names = self.dataset["train"].features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")

        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return id2label

class load_imdb():
    def __init__(self, data_name, tokenizer=None):
        self.dataset = load_dataset(data_name)
        self.tokenizer = tokenizer

    def preprocess(self, examples):
        tokenized = self.tokenizer(examples['text'], truncation=True, padding=True,)
        return tokenized

    def load_data(self, test=False):
        tokenized_dataset = self.dataset.map(self.preprocess, batched=True, remove_columns=["text"])
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['test'].shard(num_shards=2, index=0)
        test_dataset = tokenized_dataset['test'].shard(num_shards=2, index=1)

        return train_dataset, eval_dataset, test_dataset

    def id2label(self):
        num_labels = self.dataset['train'].features['label'].num_classes
        class_names = self.dataset["train"].features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")

        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return id2label

class load_SST2():
    def __init__(self, data_name,tokenizer=None):
        # self.dataset = load_dataset(data_name,split=['train', 'validation','test'])
        self.trainDataset, self.validation_Dataset, self.test_Dataset = load_dataset(data_name, split=['train', 'validation', 'test'])
        self.tokenizer = tokenizer

    def preprocess(self,batch):
        tokenized = self.tokenizer(batch['sentence'], truncation=True, padding=True)
        return tokenized

    def load_data(self,test=False):

        trainDataset = self.trainDataset.map(self.preprocess, batched=True)
        validation_Dataset = self.validation_Dataset.map(self.preprocess, batched=True)
        test_Dataset = self.test_Dataset.map(self.preprocess, batched=True)

        trainDataset = trainDataset.rename_column("label", "labels")
        validation_Dataset = validation_Dataset.rename_column("label", "labels")
        test_Dataset = test_Dataset.rename_column("label", "labels")

        trainDataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        validation_Dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_Dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return trainDataset, validation_Dataset, test_Dataset

    def id2label(self):
        num_labels = self.trainDataset.features['label'].num_classes
        class_names =  self.trainDataset.features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")

        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return id2label

class load_SST2_split():
    def __init__(self, data_name, tokenizer=None):
        # self.dataset = load_dataset(data_name,split=['train', 'validation','test'])
        self.trainDataset, self.validation_Dataset, self.test_Dataset = load_dataset(data_name, split=['train', 'validation','test'])
        self.tokenizer = tokenizer

    def preprocess(self, batch):
        tokenized = self.tokenizer(batch['sentence'], truncation=True, padding=True)
        return tokenized

    def load_data(self, test=False,train_index=0):
        self.trainDataset = self.trainDataset.select(range(0,20000))
        trainDataset = self.trainDataset.map(self.preprocess, batched=True)
        validation_Dataset = self.validation_Dataset.map(self.preprocess, batched=True)
        test_Dataset = self.test_Dataset.map(self.preprocess, batched=True)

        trainDataset = trainDataset.rename_column("label", "labels")
        validation_Dataset = validation_Dataset.rename_column("label", "labels")
        test_Dataset = test_Dataset.rename_column("label", "labels")

        trainDataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        validation_Dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_Dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return trainDataset, validation_Dataset, test_Dataset

    def load_data2(self,
                   test=False,
                   train_index=0,
                   round_sizes=[10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
                   replay_ratios=None,
                   replay_mode="partial"):
        if replay_ratios is None and replay_mode == "partial":
            replay_ratios = [0.2] * train_index

        cumulative_sizes = [sum(round_sizes[:i]) for i in range(len(round_sizes))]

        current_start = cumulative_sizes[train_index]
        current_end = cumulative_sizes[train_index + 1]
        selected_indices = list(range(current_start, current_end))  # 当前轮数据
        if replay_mode == "partial":
            for i in range(train_index):
                hist_start = cumulative_sizes[i]
                hist_end = cumulative_sizes[i + 1]
                hist_indices = list(range(hist_start, hist_end))

                num_samples = int(replay_ratios[i] * len(hist_indices))
                sampled = random.sample(hist_indices, num_samples)
                selected_indices.extend(sampled)

        elif replay_mode == "full":
            for i in range(train_index):
                hist_start = cumulative_sizes[i]
                hist_end = cumulative_sizes[i + 1]
                selected_indices.extend(range(hist_start, hist_end))

        elif replay_mode == "none":
            pass

        else:
            raise ValueError(f"Invalid replay_mode: {replay_mode}. Choose from ['partial', 'full', 'none'].")

        trainDataset = self.trainDataset.select(selected_indices)

        trainDataset = trainDataset.map(self.preprocess, batched=True)
        validation_Dataset = self.validation_Dataset.map(self.preprocess, batched=True)
        test_Dataset = self.test_Dataset.map(self.preprocess, batched=True)

        for dataset in [trainDataset, validation_Dataset, test_Dataset]:
            dataset = dataset.rename_column("label", "labels")
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return trainDataset, validation_Dataset, test_Dataset

    def id2label(self):
        num_labels = self.trainDataset.features['label'].num_classes
        class_names = self.trainDataset.features["label"].names
        print(f"number of labels: {num_labels}")
        print(f"the labels: {class_names}")

        # Create an id2label mapping
        # We will need this for our classifier.
        id2label = {i: label for i, label in enumerate(class_names)}

        return id2label

class load_CSQA():
    def __init__(self, data_name,tokenizer=None):
        self.data = load_dataset(data_name)
        self.tokenizer = tokenizer

    def process_choices(self,choices_dict):
        """
            解析 OpenBookQA 的 choices 字段，提取选项文本和对应标签。
            """
        all_labels = []
        all_texts = []

        for choices in choices_dict:
            labels = choices["label"]  # ['A', 'B', 'C', 'D']
            texts = choices["text"]  # ["straw", "Glass", "Candle", "mailing tube"]
            all_labels.append(labels)
            all_texts.append(texts)

        return all_labels, all_texts

    def preprocess_function(self,examples):
        """
        处理 OpenBookQA 数据，包括：
        1. 拆分 choices 选项
        2. 组合问题 + 选项
        3. 进行 tokenizer 处理
        """
        # 解析选项
        all_labels, all_texts = self.process_choices(examples["choices"])

        first_sentences = [[q] * len(c) for q, c in zip(examples["question"], all_texts)]
        second_sentences = all_texts  # 每个问题的四个选项

        # 展平数据
        first_sentences = sum(first_sentences, [])  # Flatten
        second_sentences = sum(second_sentences, [])  # Flatten

        # Tokenizer
        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)

        # 重新整理数据，恢复 batch 结构
        num_choices = len(all_texts[0])  # 选项数，通常是 4
        tokenized_inputs = {k: [v[i:i + num_choices] for i in range(0, len(v), num_choices)] for k, v in
                            tokenized_examples.items()}

        # 添加 labels
        answer_indices = [labels.index(ans) for labels, ans in zip(all_labels, examples["answerKey"])]
        tokenized_inputs["labels"] = answer_indices

        return tokenized_inputs

    def load_data(self,test=False):

        tokenized_train_datasets = self.data['train'].map(self.preprocess_function, batched=True)
        tokenized_val_datasets = self.data['validation'].map(self.preprocess_function, batched=True)
        return tokenized_train_datasets, tokenized_val_datasets,None

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)  # 批次大小
        num_choices = len(features[0]['input_ids'])  # 选项数
        # 将 features 中每个样本的特征扁平化为每个选项的独立特征
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        # 将扁平化后的特征列表合并成一个单一的列表，便于后续处理
        flattened_features = sum(flattened_features, [])

        # 使用 tokenizer 对合并后的特征进行填充，将其转换为统一的形状
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        # 将填充后的批次数据重塑为三维张量，形状为 (batch_size, num_choices, -1)
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics_regression(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.squeeze()
    rmse = mean_squared_error(labels, preds, squared=False)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }

def compute_metrics_cls(eval_pred):
    logits, labels = eval_pred
    # predictions = np.argmax(logits, axis=-1)  # 找到概率最高的类
    predictions = logits
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    # print(acc, precision, recall, f1)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }