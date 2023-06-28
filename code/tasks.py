# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
import numpy as np
import ast

import log
from pet import task_helpers
from pet.utils import InputExample

logger = log.get_logger('root')


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    # def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
    #     return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()


class AgnewsProcessor(DataProcessor):
    """Processor for the AG news data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class YahooAnswersProcessor(DataProcessor):
    """Processor for the Yahoo Answers data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (set_type, idx)
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class YelpPolarityProcessor(DataProcessor):
    """Processor for the YELP binary classification set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class YelpFullProcessor(YelpPolarityProcessor):
    """Processor for the YELP full classification set."""

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]


class XStanceProcessor(DataProcessor):
    """Processor for the X-Stance data set."""

    def __init__(self, language: str = None):
        if language is not None:
            assert language in ['de', 'fr']
        self.language = language

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST"]

    def _create_examples(self, path: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label']
                id_ = example_json['id']
                text_a = example_json['question']
                text_b = example_json['comment']
                language = example_json['language']

                if self.language is not None and language != self.language:
                    continue

                example = InputExample(guid=id_, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set."""

    def __init__(self):
        self.mnli_processor = MnliProcessor()

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str, hypothesis_name: str = "hypothesis",
                         premise_name: str = "premise") -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = line_idx
                label = example_json.get('label')
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class AxGProcessor(RteProcessor):
    """Processor for the AX-G diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "test")


class AxBProcessor(RteProcessor):
    """Processor for the AX-B diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "test")

    def _create_examples(self, path, set_type, hypothesis_name="sentence2", premise_name="sentence1"):
        return super()._create_examples(path, set_type, hypothesis_name, premise_name)


class CbProcessor(RteProcessor):
    """Processor for the CB data set."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]


class WicProcessor(DataProcessor):
    """Processor for the WiC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["F", "T"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = "T" if example_json.get('label') else "F"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                meta = {'word': example_json['word']}
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx, meta=meta)
                examples.append(example)
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['text']
                meta = {
                    'span1_text': example_json['target']['span1_text'],
                    'span2_text': example_json['target']['span2_text'],
                    'span1_index': example_json['target']['span1_index'],
                    'span2_index': example_json['target']['span2_index']
                }

                # the indices in the dataset are wrong for some examples, so we manually fix them
                span1_index, span1_text = meta['span1_index'], meta['span1_text']
                span2_index, span2_text = meta['span2_index'], meta['span2_text']
                words_a = text_a.split()
                words_a_lower = text_a.lower().split()
                words_span1_text = span1_text.lower().split()
                span1_len = len(words_span1_text)

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    for offset in [-1, +1]:
                        if words_a_lower[span1_index + offset:span1_index + span1_len + offset] == words_span1_text:
                            span1_index += offset

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    logger.warning(f"Got '{words_a_lower[span1_index:span1_index + span1_len]}' but expected "
                                   f"'{words_span1_text}' at index {span1_index} for '{words_a}'")

                if words_a[span2_index] != span2_text:
                    for offset in [-1, +1]:
                        if words_a[span2_index + offset] == span2_text:
                            span2_index += offset

                    if words_a[span2_index] != span2_text and words_a[span2_index].startswith(span2_text):
                        words_a = words_a[:span2_index] \
                                  + [words_a[span2_index][:len(span2_text)], words_a[span2_index][len(span2_text):]] \
                                  + words_a[span2_index + 1:]

                assert words_a[span2_index] == span2_text, \
                    f"Got '{words_a[span2_index]}' but expected '{span2_text}' at index {span2_index} for '{words_a}'"

                text_a = ' '.join(words_a)
                meta['span1_index'], meta['span2_index'] = span1_index, span2_index

                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                if set_type == 'train' and label != 'True':
                    continue
                examples.append(example)

        return examples


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = str(example_json['label']) if 'label' in example_json else None
                idx = example_json['idx']
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['premise']
                meta = {
                    'choice1': example_json['choice1'],
                    'choice2': example_json['choice2'],
                    'question': example_json['question']
                }
                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                examples.append(example)

        if set_type == 'train' or set_type == 'unlabeled':
            mirror_examples = []
            for ex in examples:
                label = "1" if ex.label == "0" else "0"
                meta = {
                    'choice1': ex.meta['choice2'],
                    'choice2': ex.meta['choice1'],
                    'question': ex.meta['question']
                }
                mirror_example = InputExample(guid=ex.guid + 'm', text_a=ex.text_a, label=label, meta=meta)
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            logger.info(f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}...")
        return examples


class MultiRcProcessor(DataProcessor):
    """Processor for the MultiRC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = str(answer_json["label"]) if 'label' in answer_json else None
                        answer_idx = answer_json["idx"]
                        guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path, set_type, seed=42, max_train_candidates_per_question: int = 10) -> List[InputExample]:
        examples = []

        entity_shuffler = random.Random(seed)

        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json['idx']
                text = example_json['passage']['text']
                entities = set()

                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = text[start:end + 1]
                    entities.add(entity)

                entities = list(entities)

                text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
                questions = example_json['qas']

                for question_json in questions:
                    question = question_json['query']
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = answer_json['text']
                        answers.add(answer)

                    answers = list(answers)

                    if set_type == 'train':
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{set_type}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta,
                                                   idx=ex_idx)
                            examples.append(example)

                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{set_type}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples

class Semeval2016t6Processor(DataProcessor):
    """Processor for the 2016t6 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(Semeval2016t6Processor._read_tsv(os.path.join(data_dir, "trainingdata-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(Semeval2016t6Processor._read_dev_tsv([os.path.join(data_dir, "trialdata-all-annotations.txt"),
                                                                           os.path.join(data_dir, "trainingdata-all-annotations.txt")],
                                                                      os.path.join(data_dir, "split/semeval2016t6_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(Semeval2016t6Processor._read_tsv(os.path.join(data_dir, "testdata-taskA-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_test.csv")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["AGAINST", "FAVOR", "NONE"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        # for i in lines:
        #     print(i)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1][1]
            text_b = line[1][0]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, split_file, quotechar=None):
        with open(input_file, "r", encoding="ISO-8859-1") as f, \
                open(split_file, "r") as split_file:
            data = list(csv.reader(f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                line.append(line_number)
                # print(data[line_number])
                line.append((data[line_number][1], data[line_number][2]))
                line.append(data[line_number][3])
                lines.append(line)


            return lines

    @staticmethod
    def _read_dev_tsv(input_file, split_file, quotechar=None):
        with open(input_file[0], "r", encoding="ISO-8859-1") as dev_f, \
                open(input_file[1], "r", encoding="ISO-8859-1") as train_f, \
                open(split_file, "r") as split_file:
            dev_data = list(csv.reader(dev_f, delimiter='\t', quotechar='"', ))
            train_data = list(csv.reader(train_f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                if data_file == 'train':
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((train_data[line_number][1], train_data[line_number][2]))
                    line.append(train_data[line_number][3])
                else:
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((dev_data[line_number][1], dev_data[line_number][2]))
                    line.append(dev_data[line_number][3])
                lines.append(line)

            return lines


class ArgminProcessor(DataProcessor):
    """Processor for the Argmin data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "train")[0], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "dev")[1], "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "test")[2], "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["Argument_against", "Argument_for"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1]
            text_b = line[0][0]
            label = line[-1]
            # print(line)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, set, quotechar=None):
        gold_files = os.listdir(input_file)
        train_lines = []
        dev_lines = []
        test_lines = []
        count = 0
        for data_file in gold_files:
            if data_file.endswith(".tsv"):
                topic = data_file.replace(".tsv", "")
                with open(input_file + data_file, 'r') as f_in:
                    reader = csv.reader(f_in, delimiter="\t", quoting=3)
                    next(reader, None)
                    for row in reader:
                        line = []
                        if row[5] != 'NoArgument':
                            if topic == "death_penalty":
                                # print(set)
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                # if len(line) != 0:
                                dev_lines.append(line)

                            elif topic == "school_uniforms" or topic == "gun_control":

                                # print(set)
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                test_lines.append(line)
                            else:
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                # if len(line) != 0:
                                train_lines.append(line)
                                # print(topic)

        # print(len(lines))
        # print(count)
        return [train_lines, dev_lines, test_lines]


class Iac1Processor(DataProcessor):
    """Processor for the Iac1 data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(Iac1Processor._read_tsv(os.path.join(data_dir))[0], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(Iac1Processor._read_tsv(os.path.join(data_dir))[1], "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(Iac1Processor._read_tsv(os.path.join(data_dir))[2], "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["anti", "pro", "other"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1]
            if not line[0][1]:
                text_a = ' '
            # if not line[0][0]:
            #     print('text_b', line[0][0])

            text_b = line[0][0]
            label = line[-1]
            # print(line)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_and_tokenize_csv(file, is_test=False):

        def get_label(id):
            return {
                0: "pro",
                1: "anti",
                2: "other"
            }[id]

        # get sample rows with discussion ids and labels
        sample_dict = defaultdict(list)
        with open(file + "author_stance.csv", "r") as in_f:
            reader = csv.reader(in_f)
            next(reader)

            topics_count = []
            for row in reader:
                label_id = np.argmax([row[3], row[4], row[5]])
                topics_count.append(row[0])
                sample_dict[row[1]].append({
                    "author": row[2],
                    "topic": row[0],
                    "label": get_label(label_id)
                })

        # get dicussion_id -> author_sentences dict
        discussion_dict = {}
        for d_id in sample_dict.keys():
            with open(file + "discussions/" + str(d_id) + ".json", "r") as in_f:
                discussion_data = json.load(in_f)
                user_posts = defaultdict(list)
                for post in discussion_data[0]:
                    user_posts[post[2]].append(post[3])
                discussion_dict[d_id] = user_posts

        # create X,y dample lists
        X = []
        y = []
        text_lens = []
        for d_id in sample_dict:
            for author_data in sample_dict[d_id]:
                text = " ".join([s for s in discussion_dict[d_id][author_data["author"]]])
                text_lens.append(len(text.split()))
                X.append((author_data["topic"], text))
                y.append(author_data["label"])

        return X, y

    @staticmethod
    def _read_tsv(input_file, quotechar=None):


        X, y = Iac1Processor._read_and_tokenize_csv(input_file)

        # split without topic leakage
        topic_set_dict = {
            "evolution": "train",
            "death penalty": "train",
            "gay marriage": "train",
            "climate change": "dev",
            "gun control": "train",
            "healthcare": "train",
            "abortion": "train",
            "existence of god": "test",
            "communism vs capitalism": "dev",
            "marijuana legalization": "test"
        }

        X_train, X_dev, X_test = [], [], []
        for i in range(len(X)):
            if topic_set_dict[X[i][0]] == "train":
                X_train.append([X[i], y[i]])

            elif topic_set_dict[X[i][0]] == "dev":
                X_dev.append([X[i], y[i]])

            else:
                X_test.append([X[i], y[i]])
        return [X_train, X_dev, X_test]

class FNC1Processor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(FNC1Processor._read_csv_from_split(self, os.path.join(data_dir), os.path.join(data_dir, "fnc1_train.csv")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(FNC1Processor._read_csv_from_split(self, os.path.join(data_dir), os.path.join(data_dir, "fnc1_dev.csv")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(FNC1Processor._read_test_csv(os.path.join(data_dir)), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["unrelated", "discuss", "agree", "disagree"]


    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _create_data(split_path, data_sents, bodies_lookup):
        X = []
        with open(split_path, "r") as in_dev_split:
            ids = ast.literal_eval(in_dev_split.readline())
            for id in ids:
                X.append([(data_sents[id][0], bodies_lookup[int(data_sents[id][1])]), data_sents[id][2]])

        return X


    def _read_csv_from_split(self, file, split_file):
        # train and dev set are not fixed, hence, read from split_file

        if "FNC-1" in file:
            sents_path = file + "train_stances.csv"
            bodies_path = file + "train_bodies.csv"
        # else:  # if ARC corpus is preprocessed
        #     sents_path = file + "arc_stances_train.csv"
        #     bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter=',', quotechar='"')
            data_bodies = csv.reader(in_bodies, delimiter=',', quotechar='"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]): tpl[1] for tpl in data_bodies}
            data_sents = list(data_sents)

        X = self._create_data(split_file, data_sents, bodies_lookup)
        return X

    @staticmethod
    def _read_test_csv(file):
        # test set is fixed, hence, read from file

        X = []
        if "FNC-1" in file:
            sents_path = file+"competition_test_stances.csv"
            bodies_path = file+"competition_test_bodies.csv"
        # else: # if ARC corpus is preprocessed
        #     sents_path = file+"arc_stances_test.csv"
        #     bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}

            # generate train instances
            for row in data_sents:
                if row[2] not in ["unrelated", "discuss", "disagree", "agree"]:
                    continue

                X.append([(row[0], bodies_lookup[int(row[1])]), row[2]])
                # y.append(row[2])

        return X

class ARCProcessor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(ARCProcessor._read_csv_from_split(self, os.path.join(data_dir), os.path.join(data_dir, "arc_train.csv")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(ARCProcessor._read_csv_from_split(self, os.path.join(data_dir), os.path.join(data_dir, "arc_dev.csv")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(ARCProcessor._read_test_csv(os.path.join(data_dir)), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["unrelated", "discuss", "agree", "disagree"]


    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _create_data(split_path, data_sents, bodies_lookup):
        X = []
        with open(split_path, "r") as in_dev_split:
            ids = ast.literal_eval(in_dev_split.readline())
            for id in ids:
                X.append([(data_sents[id][0], bodies_lookup[int(data_sents[id][1])]), data_sents[id][2]])

        return X


    def _read_csv_from_split(self, file, split_file):
        # train and dev set are not fixed, hence, read from split_file


        sents_path = file + "arc_stances_train.csv"
        bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter=',', quotechar='"')
            data_bodies = csv.reader(in_bodies, delimiter=',', quotechar='"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]): tpl[1] for tpl in data_bodies}
            data_sents = list(data_sents)

        X = self._create_data(split_file, data_sents, bodies_lookup)
        return X

    @staticmethod
    def _read_test_csv(file):
        # test set is fixed, hence, read from file

        X = []

        sents_path = file+"arc_stances_test.csv"
        bodies_path = file + "arc_bodies.csv"

        with open(sents_path, "r") as in_sents, open(bodies_path, "r") as in_bodies:

            data_sents = csv.reader(in_sents, delimiter = ',', quotechar = '"')
            data_bodies = csv.reader(in_bodies, delimiter = ',', quotechar = '"')
            next(data_bodies)
            next(data_sents)

            # generate lookup for bodies
            bodies_lookup = {int(tpl[0]):tpl[1] for tpl in data_bodies}

            # generate train instances
            for row in data_sents:
                if row[2] not in ["unrelated", "discuss", "disagree", "agree"]:
                    continue

                X.append([(row[0], bodies_lookup[int(row[1])]), row[2]])
                # y.append(row[2])

        return X

class PerspectrumProcessor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(PerspectrumProcessor._read_and_tokenize_csv(os.path.join(data_dir), 'train'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(PerspectrumProcessor._read_and_tokenize_csv(os.path.join(data_dir), 'dev'), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(PerspectrumProcessor._read_and_tokenize_csv(os.path.join(data_dir), 'test'), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["UNDERMINE", "SUPPORT"]


    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_and_tokenize_csv(file, set_type):

        with open(file + "dataset_split_v1.0.json", "r") as split_in, \
                open(file + "perspectrum_with_answers_v1.0.json", "r") as claims_in, \
                open(file + "perspective_pool_v1.0.json", "r") as perspectives_in:

            # load files
            data_split = json.load(split_in)
            claims = json.load(claims_in)
            perspectives = json.load(perspectives_in)

            # lookup for perspective ids
            perspectives_dict = {}
            for p in perspectives:
                perspectives_dict[p['pId']] = p['text']

            # init
            X = []


            # fill train/dev/test
            for claim in claims:
                cId = str(claim['cId'])
                for p_cluster in claim['perspectives']:
                    cluster_label = p_cluster['stance_label_3']
                    for pid in p_cluster['pids']:
                        if data_split[cId] == set_type:
                            X.append([(perspectives_dict[pid], claim['text']), cluster_label])
        return X

class IbmcsProcessor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(IbmcsProcessor._read(os.path.join(data_dir), 'ibmcs_train.csv'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(IbmcsProcessor._read(os.path.join(data_dir), 'ibmcs_dev.csv'), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(IbmcsProcessor._read(os.path.join(data_dir), 'ibmcs_test.csv'), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["CON", "PRO"]


    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read(file, split_file):
        X = []

        with open(file + 'claim_stance_dataset_v1.csv', "r") as in_f, open(file + split_file, "r") as split_in:
            data = csv.reader(in_f, delimiter=',', quotechar='"')
            data = list(data)

            for split_id in split_in.readlines():
                id = int(split_id.rstrip())
                X.append([(data[id][2], data[id][7]), data[id][6]])

        return X

import re

class SnopesProcessor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(SnopesProcessor._read(os.path.join(data_dir), 'train'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(SnopesProcessor._read(os.path.join(data_dir), 'dev'), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(SnopesProcessor._read(os.path.join(data_dir), 'test'), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["refute", "agree"]


    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][0]
            text_b = line[0][1]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read(file, set_type):
        data_dict = {}
        with open(file + "snopes_corpus_2.csv", "r") as in_f, open(file + "/snopes_" + set_type + ".csv",
                                                                   "r") as split_out:
            data = csv.reader(in_f, delimiter=',', quotechar='"')
            next(data)

            for row in data:
                if row[11] == "nostance" or "no_evidence" in row[12]:
                    continue

                evid_ids = row[12].split("_")[1:]
                fges = re.findall("[0-9]*_{(.+?)}", row[9])
                claim = row[6].lower()

                if len(evid_ids) == 0:
                    continue

                assert len(re.findall("[0-9]*_{(.+?)}", row[9])) == len(
                    re.findall("[0-9]*_{.+?}", row[9])), "Error in regex"

                for evid_id in evid_ids:
                    split_fge = fges[int(evid_id)].lower()

                    if row[0] not in data_dict.keys():
                        data_dict[row[0]] = {
                            "claim": claim,
                            "label": row[11],
                            "evidence": {evid_id: split_fge}
                        }
                    else:
                        data_dict[row[0]]["evidence"][evid_id] = split_fge

            X = []
            for line in split_out.readlines():
                claim_id, evid_id = line.rstrip().split(",")
                X.append([(data_dict[claim_id]['claim'], data_dict[claim_id]['evidence'][evid_id]),
                          data_dict[claim_id]["label"]])
        return X

import zipfile

class Semeval2019t7Processor(DataProcessor):
    """Processor for the fnc1 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(Semeval2019t7Processor.read_and_tokenize_zip(data_dir + "rumoureval-2019-training-data.zip", "train-key.json",
                                                                                  Semeval2019t7Processor.parse_tweets(data_dir + "rumoureval-2019-training-data.zip")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(Semeval2019t7Processor.read_and_tokenize_zip(data_dir + "rumoureval-2019-training-data.zip", "dev-key.json",
                                                                                  Semeval2019t7Processor.parse_tweets(data_dir + "rumoureval-2019-training-data.zip")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(Semeval2019t7Processor.read_and_tokenize_json(data_dir + "final-eval-key.json",
                                                                                  Semeval2019t7Processor.parse_tweets(data_dir + "rumoureval-2019-test-data.zip")), 'test')

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["support", "deny", "query", "comment"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if not line[0]:
                continue
            text_a = line[0]
            text_b = ' '
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples
    @staticmethod
    def parse_tweets(folder_path):
        # create a dict with key = reply_tweet_id and values = source_tweet_id, source_tweet_text, reply_tweet_text
        tweet_dict = {}
        with zipfile.ZipFile(folder_path, 'r') as z:
            for filename in z.namelist():
                if not filename.lower().endswith(".json") or filename.rsplit("/", 1)[1] in ['raw.json',
                                                                                            'structure.json',
                                                                                            'dev-key.json',
                                                                                            'train-key.json']:
                    continue
                with z.open(filename) as f:
                    data = f.read()
                    d = json.loads(data.decode("ISO-8859-1"))

                    if "data" in d.keys():  # reddit
                        if "body" in d['data'].keys():  # reply
                            tweet_dict[d['data']['id']] = d['data']['body']
                        elif "children" in d['data'].keys() and isinstance(d['data']['children'][0], dict):
                            tweet_dict[d['data']['children'][0]['data']['id']] = d['data']['children'][0]['data'][
                                'title']
                        else:  # source
                            try:
                                tweet_dict[d['data']['children'][0]] = ""
                            except Exception as e:
                                print(e)

                    if "text" in d.keys():  # twitter
                        tweet_dict[str(d['id'])] = d['text']

        return tweet_dict

    @staticmethod
    def read_and_tokenize_json(file_name, tweet_dict):
        X, y = [], []

        with open(file_name, "r") as in_f:
            split_dict = json.load(in_f)['subtaskaenglish']

            for tweet_id, label in split_dict.items():
                X_meta = []
                try:
                    X_meta.append(tweet_dict[tweet_id])
                except:
                    continue
                X_meta.append(label)
                X.append(X_meta)
        return X

    @staticmethod
    def read_and_tokenize_zip(folder_path, set_file, tweet_dict):
        X, y = [], []

        with zipfile.ZipFile(folder_path, 'r') as z:
            with z.open("rumoureval-2019-training-data/" + set_file) as in_f:
                split_dict = json.load(in_f)['subtaskaenglish']

                for tweet_id, label in split_dict.items():
                    X_meta = []
                    try:
                        X_meta.append(tweet_dict[tweet_id])
                    except:
                        continue
                    X_meta.append(label)
                    X.append(X_meta)

        return X

class PstanceProcessor(DataProcessor):
    def get_train_examples(self, data_dir):

        return self._create_examples(PstanceProcessor.sample_incorporate(data_dir + 'raw_train_'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(PstanceProcessor.sample_incorporate(data_dir + 'raw_val_'), 'dev')
    def get_test_examples(self, data_dir):
        return self._create_examples(PstanceProcessor.sample_incorporate(data_dir + 'raw_test_'), 'test')
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST", "NONE"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        # print(len(lines))
        for (i, line) in enumerate(lines):
            # print(i, line)
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # if not line[0]:
            #     continue
            text_a = line[1][1]
            text_b = line[1][0]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            # print(example)
            examples.append(example)
        # print(examples[-1])
        return examples

    @staticmethod
    def sample_incorporate(path):
        bernie_path = path + 'bernie.csv'
        biden_path = path + 'biden.csv'
        trump_path = path + 'trump.csv'
        lines = []
        count = 0
        with open(bernie_path, 'r') as f:
            bernie_data = list(csv.reader(f))
            for i in bernie_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1

        with open(biden_path, 'r') as f:
            biden_data = list(csv.reader(f))
            for i in biden_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1
        with open(trump_path, 'r') as f:
            trump_data = list(csv.reader(f))
            for i in trump_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1
        return lines


# tweet id --- no content
class WtwtProcessor(DataProcessor):
    def get_train_examples(self, data_dir):

        return self._create_examples(WtwtProcessor.load_json(data_dir + 'wtwt_ids.json'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(WtwtProcessor.sample_incorporate(data_dir + 'raw_val_'), 'dev')
    def get_test_examples(self, data_dir):
        return self._create_examples(WtwtProcessor.sample_incorporate(data_dir + 'raw_test_'), 'test')
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST", "NONE"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        # print(len(lines))
        for (i, line) in enumerate(lines):
            # print(i, line)
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # if not line[0]:
            #     continue
            text_a = line[1][1]
            text_b = line[1][0]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            # print(example)
            examples.append(example)
        # print(examples[-1])
        return examples

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            data = f.readlines()
            for i in data:
                print(i)

        print(data)

    @staticmethod
    def sample_incorporate(path):
        bernie_path = path + 'bernie.csv'
        biden_path = path + 'biden.csv'
        trump_path = path + 'trump.csv'
        lines = []
        count = 0
        with open(bernie_path, 'r') as f:
            bernie_data = list(csv.reader(f))
            for i in bernie_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1

        with open(biden_path, 'r') as f:
            biden_data = list(csv.reader(f))
            for i in biden_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1
        with open(trump_path, 'r') as f:
            trump_data = list(csv.reader(f))
            for i in trump_data[1:]:
                line = []
                line.append(count)
                content = (i[0], i[1])
                label = i[2]
                line.append(content)
                line.append(label)
                lines.append(line)
                count += 1
        return lines


PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "xstance": XStanceProcessor,
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
    "ax-g": AxGProcessor,
    "ax-b": AxBProcessor,
    "2016t6": Semeval2016t6Processor,
    "argmin": ArgminProcessor,
    "iac1": Iac1Processor,
    "fnc1": FNC1Processor,
    "perspectrum": PerspectrumProcessor,
    "ibmcs": IbmcsProcessor,
    "snopes": SnopesProcessor,
    "2019t7": Semeval2019t7Processor,
    "arc": ARCProcessor,
    "pstance": PstanceProcessor,
    "wtwt": WtwtProcessor,
    }  # type: Dict[str,Callable[[],DataProcessor]]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"]
}

DEFAULT_METRICS = ["acc", "f1-macro", "all"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
        # print(examples)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")
    # for i in examples:
    #     print(i)
    if num_examples is not None:
        # print(examples)
        # for i in examples:
        #     print(i)
        # print(examples)
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples

if __name__ == '__main__':
    train_data = load_examples(
        'pstance', '../GLUE_data/PStance/', TRAIN_SET, num_examples=-1)
    for i in train_data:
        print(i)