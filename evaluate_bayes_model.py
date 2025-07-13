# bayes_utils.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from load_data import compute_metrics_cls
from sklearn.metrics import accuracy_score
from datasets import load_metric
metric = load_metric('/accuracy.py')

def evaluate_model(inference_model, dataset,data_collator=None):
    eval_dataloader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()
    # inference_model.train()
    all_predictions = []
    all_labels = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        all_predictions.append(predictions)
        all_labels.append(batch["labels"])

    eval_metric = metric.compute()
    print(eval_metric)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    print(compute_metrics_cls((all_predictions.cpu().numpy(), all_labels.cpu().numpy())))


batch_accuracies = []
batch_uncertainties = []

#
def evaluate_model_bayes(inference_model, dataset, num_samples=10, uncertainty_measure='variance', num_bins=100,
                         rejection_plot=False,data_collator=None):
    eval_dataloader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    all_labels = []
    all_uncertainties = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        probabilities = []

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = inference_model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)  # 获取每个类别的概率
                probabilities.append(probs)

        probabilities = torch.stack(probabilities)  # [num_samples, batch_size, num_classes]

        mean_probs = probabilities.mean(dim=0)  # [batch_size, num_classes]
        var_probs = probabilities.var(dim=0)  # [batch_size, num_classes]

        predictions = mean_probs.argmax(dim=-1)

        if uncertainty_measure == "mi":
            # --- 互信息计算 ---
            entropy_mean_probs = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)  # 平均概率的熵
            entropy_sample_probs = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1).mean(
                dim=0)  # 每次采样熵的平均值
            uncertainties = entropy_mean_probs - entropy_sample_probs  # 互信息

        elif uncertainty_measure == "kl":
            # --- KL 散度计算 ---
            kl_divergences = []
            for i in range(num_samples):
                kl_div = torch.sum(
                    probabilities[i] * (torch.log(probabilities[i] + 1e-10) - torch.log(mean_probs + 1e-10)), dim=-1)
                kl_divergences.append(kl_div)
            uncertainties = torch.stack(kl_divergences).mean(dim=0)  # 平均 KL 散度作为不确定性

        elif uncertainty_measure == "variance":
            uncertainties = probabilities.var(dim=0).sum(dim=-1)

        all_uncertainties.append(uncertainties)
        all_predictions.append(predictions)
        all_labels.append(batch["labels"])

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_uncertainties = torch.cat(all_uncertainties)

    if rejection_plot:
        rejection_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rejection_accuracies = []

        for rejection_threshold in rejection_ratios:
            threshold = torch.quantile(all_uncertainties, 1 - rejection_threshold)

            keep_mask = all_uncertainties <= threshold
            filtered_predictions = all_predictions[keep_mask]
            filtered_labels = all_labels[keep_mask]

            metric.add_batch(
                predictions=filtered_predictions,
                references=filtered_labels,
            )
            eval_metric = metric.compute()

            accuracy = eval_metric['accuracy']
            rejection_accuracies.append(accuracy)

            print(f"Evaluation after rejecting top {100 * rejection_threshold}% uncertain samples:")
            print(eval_metric)
        x_values = [100 * ratio for ratio in rejection_ratios]
        y_values = rejection_accuracies
        plt.figure(figsize=(8, 6))

        plt.plot( x_values, y_values, label=None,marker='o',markersize=10,linewidth=3)

        df = pd.DataFrame({'Rejection Percentage (%)': x_values, 'Accuracy': y_values})

        csv_filename = "accuracy_vs_rejection_SST2.csv"
        df.to_csv(csv_filename, index=False)

        plt.xlabel("Rejection Percentage (%)", fontsize=20,fontweight='bold')  # X 轴标签
        plt.ylabel("Accuracy", fontsize=20,fontweight='bold')  # Y 轴标签
        plt.title("Accuracy Rejection Curve on SST2", fontsize=22,fontweight='bold')  # 标题

        plt.margins(x=0.01, y=0.01)
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

        plt.xticks(fontsize=18,fontweight='bold')
        plt.yticks(fontsize=18,fontweight='bold')
        # plt.legend(fontsize=14)
        # plt.legend(fontsize=18, prop={'weight': 'bold'})
        plt.grid(True)

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig("./accuracy_vs_rejection.svg", format="svg", dpi=300)
        plt.show()
    metric.add_batch(
        predictions=all_predictions,
        references=all_labels,
    )
    eval_metric = metric.compute()
    print(eval_metric)
    print(compute_metrics_cls((all_predictions.cpu().numpy(), all_labels.cpu().numpy())))
    return var_probs
