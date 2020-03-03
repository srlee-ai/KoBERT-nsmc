import argparse
import json

from trainer import Trainer
from predict import Predict
from utils import init_logger, load_tokenizer, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    tokenizer = load_tokenizer(args)
    train_dataset = None if args.do_predict else load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = None
    test_dataset = None if args.do_predict else load_and_cache_examples(args, tokenizer, mode="test")

    if args.do_train:
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
        trainer.train()

    if args.do_eval:
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
        trainer.load_model()
        trainer.evaluate("test")

    if args.do_predict:
        predict = Predict(args, tokenizer)
        predict.load_model()

        sentences = [args.sentence]
        result_json = dict()
        result_json['result'] = int(predict.predict(sentences))
        print(json.dumps(result_json, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="ratings_train.txt", type=str, help="Train file")
    parser.add_argument("--test_file", default="ratings_test.txt", type=str, help="Test file")
    parser.add_argument("--sentence", default="연기는 별로지만 재미 하나는 끝내줌!", type=str, help="predict sentence")

    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=2000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict senctence.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
