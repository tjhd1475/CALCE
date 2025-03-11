"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.moment_retrieval_dataset import MomentRetrievalDataset
from lavis.datasets.datasets.mr_stage1_dataset import MRStage1Dataset
from lavis.datasets.datasets.mr_stage2_dataset import MRStage2Dataset


class MomentRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = MomentRetrievalDataset
    eval_dataset_cls = MomentRetrievalDataset

    def build(self):
        datasets = super().build()

        return datasets


@registry.register_builder("qvh")
class QVHBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvh/defaults.yaml",
    }


@registry.register_builder("charades_sta")
class Charades_STABuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/defaults.yaml",
    }


@registry.register_builder("anet")
class ANetBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anet/defaults.yaml",
    }


@registry.register_builder("tacos")
class TACoSBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/tacos/defaults.yaml",
    }

#########################################################

class MRStage1Builder(BaseDatasetBuilder):
    train_dataset_cls = MRStage1Dataset
    eval_dataset_cls = MRStage1Dataset

    def build(self):
        datasets = super().build()

        return datasets

@registry.register_builder("qvh_stage1")
class QVHStage1Builder(MRStage1Builder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvh/defaults_stage1.yaml",
    }

@registry.register_builder("charades_sta_stage1")
class CharadesStage1Builder(MRStage1Builder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/defaults_stage1.yaml",
    }

class MRStage2Builder(BaseDatasetBuilder):
    train_dataset_cls = MRStage2Dataset
    eval_dataset_cls = MRStage2Dataset

    def build(self):
        datasets = super().build()

        return datasets

@registry.register_builder("qvh_stage2")
class QVHStage2Builder(MRStage2Builder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvh/defaults_stage2.yaml",
    }

@registry.register_builder("charades_sta_stage2")
class CharadesStage2Builder(MRStage2Builder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charades_sta/defaults_stage2.yaml",
    }



# open-ended QA
