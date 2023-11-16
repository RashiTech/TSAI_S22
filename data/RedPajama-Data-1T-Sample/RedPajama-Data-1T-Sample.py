# Copyright 2023 Together Computer
#
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

# Lint as: python3
"""RedPajama: An Open-Source, Clean-Room 1.2 Trillion Token Dataset."""


import json

import datasets


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset. This is a 1B-token sample of the full dataset.
"""

_URLS = [
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
    "c4_sample.jsonl",
    "cc_2019-30_sample.jsonl",
    "cc_2020-05_sample.jsonl",
    "cc_2021-04_sample.jsonl",
    "cc_2022-05_sample.jsonl",
    "cc_2023-06_sample.jsonl",
    "github_sample.jsonl",
    "stackexchange_sample.jsonl",
    "wikipedia_sample.jsonl",
]


class RedPajama1TSampleConfig(datasets.BuilderConfig):
    """BuilderConfig for RedPajama sample."""

    def __init__(self, **kwargs):
        """BuilderConfig for RedPajama sample.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RedPajama1TSampleConfig, self).__init__(**kwargs)


class RedPajama1TSample(datasets.GeneratorBasedBuilder):
    """RedPajama 1T Sample: version 1.0.0."""

    BUILDER_CONFIGS = [
        RedPajama1TSampleConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": downloaded_files})
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepaths)
        key = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    if "meta" not in data:
                        text = data["text"]
                        del data["text"]
                        yield key, {
                            "text": text,
                            "meta": json.dumps(data),
                        }
                    else:
                        yield key, {
                            "text": data["text"],
                            "meta": data["meta"],
                        }
                    key += 1
