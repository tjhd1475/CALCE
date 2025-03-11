import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def _loss_saliency(self, outputs):
        # target_saliency_score: b, t
        # saliency_score: b, t
        saliency_score, saliency_score_neg, target_saliency_score = outputs['pred_saliency_score'].clone(), outputs['pred_saliency_score_neg'].clone(), outputs['target_saliency_score']

        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_score_neg))).sum(dim=1).mean()

        saliency_score = torch.cat([saliency_score,saliency_score_neg],dim=1)
        target_saliency_score = torch.cat([target_saliency_score,torch.zeros_like(target_saliency_score)],dim=1)
        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(target_saliency_score > 100)  # no drop
            pos_mask = (target_saliency_score >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_score * drop_mask / tau + ~drop_mask * -1e+3  # suppress too high score

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12
        return loss_rank_contrastive + loss_neg_pair

    def forward(self, output):
        loss_rank_contrastive = self._loss_saliency(output)
        return loss_rank_contrastive