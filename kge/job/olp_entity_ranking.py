import math
import time
from operator import itemgetter

import torch
import kge.job
from kge import Config, Dataset
from kge.job import EvaluationJob, Job, EntityRankingJob
from collections import defaultdict

class OLPEntityRankingJob(EntityRankingJob):
    """
    OLP Entity / Mention ranking evaluation protocol
    Overwrites the functions in that Mention Ranking differs from Entity Ranking.
    """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == OLPEntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        """ Uses indexes in the dataloader to be able to fetch corresponding alternative mentions. """

        self.loader = torch.utils.data.DataLoader(
            range(self.triples.shape[0]),
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _collate(self, batch):
        "Looks up true triples and the corresponding alternative mentions for each triple in the batch"
        split = self.config.get("eval.split")

        label_coords = []

        batch_data = torch.index_select(self.triples, 0, torch.tensor(batch))

        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch_data,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_sp_to_o"),
                self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch_data,
                self.dataset.num_entities(),
                self.dataset.index("test_sp_to_o"),
                self.dataset.index("test_po_to_s"),
            )
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        if self.batch_size != 1:
            alternative_subject_mentions = torch.cat(tuple(
                itemgetter(*batch)(self.dataset._alternative_subject_mentions[split])))
            alternative_object_mentions = torch.cat(
                tuple(itemgetter(*batch)(self.dataset._alternative_object_mentions[split])))
        else:
            alternative_subject_mentions = itemgetter(*batch)(self.dataset._alternative_subject_mentions[split])
            alternative_object_mentions = itemgetter(*batch)(self.dataset._alternative_object_mentions[split])

        return batch_data, label_coords, test_label_coords, alternative_subject_mentions, alternative_object_mentions

    def compute_true_scores(self, batch_coords):
        """
        Computes true scores for batch and returns the corresponding entity.
        """
        alternative_subject_mentions = batch_coords[3].to(self.device)
        alternative_object_mentions = batch_coords[4].to(self.device)
        o_true_scores_all_mentions = self.model.score_spo(alternative_object_mentions[:, 0],
                                                         alternative_object_mentions[:, 1],
                                                         alternative_object_mentions[:, 3], "o").view(-1)
        s_true_scores_all_mentions = self.model.score_spo(alternative_subject_mentions[:, 3],
                                                         alternative_subject_mentions[:, 1],
                                                         alternative_subject_mentions[:, 2], "s").view(-1)

        # inspired by https://github.com/pytorch/pytorch/issues/36748#issuecomment-620279304
        def filter_mention_results(scores, quadruples):
            ranks = torch.unique_consecutive(quadruples[:, 0:3], dim=0, return_inverse=True)[1]
            true_scores = torch.ones(ranks.max() + 1, dtype=scores.dtype, device=self.device) * float("-inf")
            true_entities = torch.zeros(ranks.max() + 1)

            for i, rank in enumerate(ranks):
                if scores[i] > true_scores[rank]:
                    true_scores[rank] = scores[i]
                    true_entities[rank] = quadruples[i, 3]

            return true_scores, true_entities

        o_true_scores, o_true_entities = filter_mention_results(o_true_scores_all_mentions, alternative_object_mentions)
        s_true_scores, s_true_entities = filter_mention_results(s_true_scores_all_mentions, alternative_subject_mentions)
        return o_true_scores, s_true_scores, o_true_entities, s_true_entities
