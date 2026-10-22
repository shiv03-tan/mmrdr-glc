"""
Grade-lesion consistency loss (rung 2 of the experimental ladder).

Drop this into train_uwf_multitask.py (or import it) and wire it into the
training loop where grade_logits and lesion_logits are already computed.

Matches the locked protocol and the block diagram already sent:
    u = 1 - g0                (P(any DR) from the grade head)
    v = g4                    (P(grade 4) from the grade head)
    a = 1 - prod(1 - p_lesion_i)              over all 7 lesions
    b = 1 - prod(1 - p_lesion_i)              over NV, VH, RD only
    consistency = JS(a, u) + JS(b, v)         (Bernoulli Jensen-Shannon)
    total_loss = grade_loss + lesion_loss + lambda(t) * consistency

Lesion order assumed everywhere: [MA, HE, IH, VB_IRMA, NV, VH, RD]
"""

import torch
import torch.nn.functional as F

LESION_ORDER = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
ADVANCED_IDX = [LESION_ORDER.index(name) for name in ("NV", "VH", "RD")]  # [4, 5, 6]


def bernoulli_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Elementwise Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q), in nats."""
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    m = 0.5 * (p + q)

    def kl_bernoulli(p1, p2):
        return p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))

    return 0.5 * kl_bernoulli(p, m) + 0.5 * kl_bernoulli(q, m)


def compute_consistency_loss(
    grade_logits: torch.Tensor,      # (B, 5) raw logits, classes 0..4
    lesion_logits: torch.Tensor,     # (B, 7) raw logits, order = LESION_ORDER
    consistency_mask: torch.Tensor = None,  # (B,) bool; True = include sample. Use this
                                             # to implement the uncertain-label policy
                                             # (e.g. exclude tr000215.jpg from this term
                                             # only, while it still contributes to
                                             # grade_loss and the other 6 lesion targets).
    eps: float = 1e-6,
) -> torch.Tensor:
    grade_probs = F.softmax(grade_logits, dim=1)   # (B, 5)
    lesion_probs = torch.sigmoid(lesion_logits)    # (B, 7)

    g0 = grade_probs[:, 0]
    g4 = grade_probs[:, 4]
    u = 1.0 - g0
    v = g4

    one_minus_all = torch.clamp(1.0 - lesion_probs, eps, 1.0)
    a = 1.0 - torch.prod(one_minus_all, dim=1)

    one_minus_adv = torch.clamp(1.0 - lesion_probs[:, ADVANCED_IDX], eps, 1.0)
    b = 1.0 - torch.prod(one_minus_adv, dim=1)

    per_sample = bernoulli_js_divergence(a, u, eps) + bernoulli_js_divergence(b, v, eps)  # (B,)

    if consistency_mask is not None:
        if consistency_mask.sum() == 0:
            # No eligible samples in this batch -> contributes nothing, but stays
            # differentiable-shaped so it can still be added to total_loss safely.
            return per_sample.sum() * 0.0
        per_sample = per_sample[consistency_mask]

    return per_sample.mean()


def consistency_weight_at_epoch(epoch: int, ramp_epochs: int, lambda_max: float) -> float:
    """Linear ramp: 0 at epoch 0 -> lambda_max at epoch >= ramp_epochs."""
    if ramp_epochs <= 0:
        return lambda_max
    return lambda_max * min(1.0, epoch / ramp_epochs)


# ---------------------------------------------------------------------------
# Integration sketch — adapt names to match train_uwf_multitask.py exactly.
# ---------------------------------------------------------------------------
#
# In the training loop, after the forward pass that already produces
# grade_logits and lesion_logits:
#
#     lam = consistency_weight_at_epoch(
#         epoch=epoch,
#         ramp_epochs=args.consistency_ramp_epochs,   # e.g. 5, matching the
#                                                       # existing 5-epoch warmup
#         lambda_max=args.consistency_weight,          # CLI arg, already exists
#     )
#
#     consistency_loss = compute_consistency_loss(
#         grade_logits, lesion_logits, consistency_mask=batch["consistency_mask"]
#     )
#
#     total_loss = grade_loss + lesion_loss + lam * consistency_loss
#
# Log both the raw (unweighted) consistency_loss and lam separately, e.g.
#     train_consistency_loss=consistency_loss.item()
#     train_consistency_weight=lam
# so a rung-2 log line is distinguishable from a rung-1 line where lam=0
# (in rung 1 the raw term can still be nonzero even though its contribution
# to total_loss is zero — that is expected and fine).
#
# consistency_mask construction: a (B,) bool tensor built from the dataset's
# per-sample uncertain-label flag (the same flag already used to mask
# tr000215.jpg out of the IH lesion target). True = eligible for the
# consistency term. This must NOT be the same as excluding a sample from
# grade_loss/lesion_loss — those keep using their own existing masks.
