import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import uvicorn
from datasets import load_dataset, load_metric
from datasets.arrow_dataset import Dataset
from fastapi import FastAPI, Request
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

app = FastAPI()
MAX_LEN = 62
logger.info("Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-xsum", do_lower_case=True
)

logger.info("Loading BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    source_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."}
    )
    target_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            pass
        #             raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("summarization") and not self.task.startswith(
            "translation"
        ):
            raise ValueError(
                "`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def _preprocess_function(examples):
    inputs = examples["review_body"]
    targets = examples["review_body"]
    inputs = [" " + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=128, padding="max_length", truncation=True
        )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@app.post("/predict")
async def index(request: Request):
    request_content_type = request.headers["Content-Type"]
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = await request.json()
        logger.info("================ input sentences ===============")

        logger.info(f"data: {data}")
        logger.info(f"data: {data['review_body']}")

        test_file = dict()
        test_file = {
            "review_id": "",
            "product_id": "",
            "reviewer_id": "",
            "stars": "",
            "review_body": data["review_body"],
            "review_title": "",
            "language": "en",
            "product_category": "",
        }

        with open(
            os.path.dirname(__file__) + "request_body.json",
            mode="wt",
            encoding="utf-8",
        ) as file:
            json.dump(test_file, file, ensure_ascii=False, indent=2)

        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_json_file(
            json_file="./args.json"
        )

        logger.info(f"training=args: {training_args}")

        # dataset = Dataset.from_dict(data)
        data_files = {}
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files)
        logger.info(f"dataset: {dataset}")

        test_dataset = dataset["test"]

        test_dataset = test_dataset.map(
            _preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=[
                "review_id",
                "product_id",
                "reviewer_id",
                "stars",
                "review_body",
                "review_title",
                "language",
                "product_category",
            ],
            load_from_cache_file=True,
        )
        logger.info(f"test_dataset: {test_dataset}")

        preds = predict(test_dataset, model)
        return preds[0]
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict(input_data, model):
    logger.info("Proccessing predict...")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_json_file(
        json_file="./args.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info(f"input_data: {input_data}")

    test_results = trainer.predict(
        input_data,
        metric_key_prefix="test",
        max_length=1024,
        num_beams=6,
    )

    logger.info(f"test_results: {test_results}")

    test_preds = tokenizer.batch_decode(
        test_results.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    logger.info(f"test_preds: {test_preds}")

    test_preds = [pred.strip() for pred in test_preds]
    logger.info(test_preds)
    return test_preds


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
