# A Bayesian Hybrid Parameter-Efficient Fine-Tuning Method for Large Language Models
This is our implementation for the paper:  

**A Bayesian Hybrid Parameter-Efficient Fine-Tuning Method for Large Language Models**.  




## Environment Settings  

- Python 3.9+
- PyTorch 2.10+
- `transformers` (HuggingFace) ≥ 4.30
- `datasets` (HuggingFace)
- `scikit-learn`  
- `accelerate`   
- `matplotlib`
- `numpy`
- `(etc.)`
## Example to Run the Codes  

Run the following command to train the model on the sst2 dataset:  

```bash
python train.py \
  --dataset_name sst2 \
  --task_type classification \
  --base_model /FacebookAI/roberta-base \
  --data_base /my_dataset \
  --lr 5e-5 \
  --batch_size 8 \
  --epochs 50 \
  --fp16 \
  --device cuda:0 \
  --adapter_name BH-PEFT \
  --prefix_bottleneck_size 8 \
  --bn_reduction_factor 96 \
  --kl_weight 0.001 \
  --prior_mu 0.0 \
  --prior_sigma 0.1 \

```
## Dataset

The datasets used in our experiments are all publicly available. You can download or load them using the HuggingFace `datasets` library or through the links below.

### CommonsenseQA (CSQA)

- **Download:** https://huggingface.co/datasets/tau/commonsense_qa

- **Description:** A multiple-choice question answering dataset that requires reasoning over commonsense knowledge. Each question has five answer options with only one correct.

- **Task Type:** Multiple-Choice QA

### Stanford Sentiment Treebank (SST-2)

- **Download:** https://huggingface.co/datasets/glue/viewer/sst2

- **Description:** Binary sentiment classification from movie reviews, widely used for sentence-level sentiment analysis.

- **Task Type:** Sentiment Classification

### AG News

- **Download:** https://huggingface.co/datasets/fancyzhx/ag_news

- **Description:** Four-class news topic classification dataset: World, Sports, Business, Sci/Tech.

- **Task Type:** Topic Classification

### IMDB
- **Download:**  https://huggingface.co/datasets/stanfordnlp/imdb

- **Description:** Binary sentiment classification on movie reviews.

- **Task Type:** Sentiment Classification

### Drug Review Dataset
- **Download:** https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com

- **Description:**  A dataset containing patient drug reviews, with numeric ratings for effectiveness, side effects, and satisfaction. In our setting, we formulate it as a regression task to predict the numeric rating.

- **Task Type:** Text Regression