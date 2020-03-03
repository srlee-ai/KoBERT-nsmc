import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences

from utils import set_seed, compute_metrics, get_label, MODEL_CLASSES

logger = logging.getLogger(__name__)


class Predict(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels, finetuning_task=args.task)
        self.model = self.model_class(self.bert_config, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    # 입력 데이터 변환
    def convert_input_data(self, sentences):
        # 문장을 토큰으로 분리
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        logger.info(sentences[0])
        logger.info(tokenized_texts[0])

        # 입력 토큰의 최대 시퀀스 길이
        MAX_LEN = 512

        # 토큰을 숫자 인덱스로 변환
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        logger.info(input_ids[0])

        # 어텐션 마스크 초기화
        attention_masks = []

        # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
        # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)

        logger.info(attention_masks)

        # 데이터를 파이토치의 텐서로 변환
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor([0])

        logger.info(inputs[0])
        logger.info(masks[0])
        logger.info(labels[0])

        return inputs, masks, labels

    def predict(self, sentences):
        logger.info(sentences)
        sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
        logger.info(sentences)

        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:', torch.cuda.get_device_name(0))

        self.model.eval()

        # 문장을 입력 데이터로 변환
        inputs, masks, labels = self.convert_input_data(sentences)

        # 데이터를 GPU에 넣음
        b_input_ids = inputs.to(self.device)
        b_input_mask = masks.to(self.device)
        b_labels = labels.to(self.device)

        # 그래디언트 계산 안함
        with torch.no_grad():
            inputs = {'input_ids': b_input_ids,
                      'attention_mask': b_input_mask,
                      'labels': b_labels}
            if self.args.model_type != 'distilkobert':
                inputs['token_type_ids'] = None
            outputs = self.model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        logger.info(logits)

        return np.argmax(logits)

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
