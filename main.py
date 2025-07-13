
import adapters
from adapters import ConfigUnion, ParBnConfig, PrefixTuningConfig
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_squared_error, mean_absolute_error, \
    r2_score
from transformers import  RobertaTokenizer, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, RobertaConfig, RobertaForSequenceClassification
from transformers import TrainingArguments
from adapters import AdapterTrainer
from load_data import load_imdb, load_CSQA, load_Ag_news, load_SST2,load_drug_reviews
from datasets import load_dataset
import loss
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian PEFT training script")

    parser.add_argument('--dataset_name', type=str, default='sst2',
                        choices=['sst2', 'ag_news', 'drug_reviews', 'csqa', 'imdb'],
                        help='Name of the dataset to train on')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'multiple_choice','regression'],
                        help='Task type')
    parser.add_argument('--save_time', type=int, default=0, help='Save index/time step')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='KL divergence loss weight')

    parser.add_argument('--base_model', type=str, default='/data/liuyang/myproject/peft-main/RoBERTa_cls/FacebookAI/roberta-base',
                        help='Path or model hub name of the base model')
    parser.add_argument('--data_base', type=str, default='/data/liuyang/myproject/BH-PEFT/my_dataset',
                        help='Root path to all datasets')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--eval_steps', type=int, default=5000, help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save checkpoint frequency')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--device', type=str, default='auto', help='CUDA device id, e.g. cuda:0')
    parser.add_argument('--adapter_name', type=str, default='BH-PEFT',
                        help='Name of the adapter being trained')
    parser.add_argument('--prefix_bottleneck_size', type=int, default=8,
                        help='Bottleneck size for PrefixTuning')
    parser.add_argument('--bn_reduction_factor', type=int, default=96,
                        help='Reduction factor for Bottleneck Adapter (ParBnConfig)')
    parser.add_argument('--do_bayesian_eval', action='store_true',
                        help='Whether to run Bayesian uncertainty evaluation after training')
    parser.add_argument('--prior_mu', type=float, default=0.0, help='Prior mean for BayesLinear')
    parser.add_argument('--prior_sigma', type=float, default=0.1, help='Prior std for BayesLinear')
    return parser.parse_args()

def load_dataset(dataset_name, tokenizer, data_base):
    path = os.path.join(data_base, dataset_name)
    if dataset_name == 'sst2':
        return load_SST2(path, tokenizer)
    elif dataset_name == 'ag_news':
        return load_Ag_news(path, tokenizer)
    elif dataset_name == 'drug_reviews':
        return load_drug_reviews(path, tokenizer)
    elif dataset_name == 'csqa':
        return load_CSQA(path, tokenizer)
    elif dataset_name == 'imdb':
        return load_imdb(path, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def build_model(args, id2label, model_class, config=None):
    if args.task_type in ["regression",'multiple_choice']:
        model = model_class.from_pretrained(
            args.base_model,
            config=config,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map=args.device,
        )
    else:
        model = model_class.from_pretrained(
            args.base_model,
            id2label=id2label,
            config=config,
            trust_remote_code=True,
            load_in_8bit=False,
            device_map=args.device,
        )
    adapters.init(model)
    config_union = ConfigUnion(
        PrefixTuningConfig(bottleneck_size=args.prefix_bottleneck_size,prior_mu=args.prior_mu,prior_sigma=args.prior_sigma),
        ParBnConfig(reduction_factor=args.bn_reduction_factor,prior_mu=args.prior_mu,prior_sigma=args.prior_sigma),
    )
    model.add_adapter(args.adapter_name, config=config_union)
    model.train_adapter(args.adapter_name)
    return model


def compute_metrics_cls(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

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
class ModifiedTrainer(AdapterTrainer):
    def __init__(self, *args, kl_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        kl = loss.BKLLoss(reduction='mean', last_layer_only=False)(model)
        ce = outputs.loss
        cost = ce + self.kl_weight * kl
        return (cost.squeeze(), outputs) if return_outputs else cost.squeeze()


def train_and_evaluate(args):
    exp_dir = f'./experiments/{args.dataset_name}_train'
    log_dir = os.path.join(exp_dir, f'{args.adapter_name}-log-{args.dataset_name}{args.save_time}')
    adapter_save_path = os.path.join(exp_dir, f'{args.adapter_name}-peft-{args.dataset_name}{args.save_time}')
    tokenizer_save_path = os.path.join(exp_dir, f'{args.adapter_name}-tokenizer-{args.dataset_name}')

    tokenizer = RobertaTokenizer.from_pretrained(args.base_model, trust_remote_code=True, model_max_length=482)
    dataset = load_dataset(args.dataset_name, tokenizer, args.data_base)
    train_dataset, eval_dataset, test_dataset = dataset.load_data()
    id2label = None
    config = None
    if args.task_type == 'multiple_choice':
        from transformers import AutoModelForMultipleChoice
        from load_data import DataCollatorForMultipleChoice
        model_class = AutoModelForMultipleChoice
        data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
        compute_metrics_fn = compute_metrics_cls
        # id2label = dataset.id2label()
    elif args.task_type == 'regression':
        config = RobertaConfig.from_pretrained(args.base_model, num_labels=1)
        model_class = RobertaForSequenceClassification
        compute_metrics_fn = compute_metrics_regression
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    else:
        model_class = AutoModelForSequenceClassification
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        compute_metrics_fn = compute_metrics_cls
        id2label = dataset.id2label()
    model = build_model(args, id2label=id2label, model_class=model_class,config=config)


    training_args = TrainingArguments(
        output_dir=log_dir,
        evaluation_strategy='steps',
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        fp16=args.fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        kl_weight=args.kl_weight
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params}, Trainable: {trainable_params}, "
          f"Percentage: {100 * trainable_params / total_params:.2f}%")

    model.is_parallelizable = True
    model.model_parallel = True

    trainer.train()

    tokenizer.save_pretrained(tokenizer_save_path)
    model.save_adapter(adapter_save_path, adapter_name=args.adapter_name, with_head=True)
    trainer.save_model()

    if args.do_bayesian_eval:
        print('Running Bayesian uncertainty estimation on test set...')
        from evaluate_bayes_model import evaluate_model_bayes
        evaluate_model_bayes(
            inference_model=model,
            dataset=test_dataset,
            num_samples=5,
            uncertainty_measure='variance',
            num_bins=100,
            rejection_plot=True,
            data_collator=data_collator,
        )
    # print('Evaluating on validation...')
    # eval_ = trainer.evaluate()
    # print(eval_)
    #
    # print('Evaluating on test...')
    # trainer = AdapterTrainer(
    #     model=model,
    #     args=training_args,
    #     eval_dataset=test_dataset,
    #     compute_metrics=compute_metrics
    # )
    # test_ = trainer.evaluate()
    # print(test_)

def main():
    args = parse_args()
    train_and_evaluate(args)

if __name__ == '__main__':
    main()