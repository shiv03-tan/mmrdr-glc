"""
Joint grade-lesion output layer for MMRDR UWF (5 grades x 7 binary lesions = 640 states).

Instead of two independent heads, the network scores every joint state
s = (grade g, lesion vector l) and normalizes over all states. Grade and lesion
probabilities are then marginals of one distribution, so they cannot contradict
each other more than the constraint mode allows.

Score of state s for image i (all terms predicted from the shared feature h_i):
    E_i(s) = u_i[g]                          grade term
           + sum_c l_c * v_i[c]              lesion terms
           + sum_c l_c * W_i[g, c]           grade-lesion interaction
           + sum_{c<d} l_c * l_d * P_i[c,d]  lesion-lesion interaction
Every term is linear in a 68-dim vector z_i = Linear(h_i), so E_i = z_i @ Phi^T
with a fixed 640 x 68 feature matrix Phi. Cost is one small matmul.

Constraint modes (annotation rules from the MMRDR grading protocol):
    R1: grade 0  <=> no lesion recorded
    R2: grade 4  <=> at least one of NV, VH, RD
    'none' : no constraint (unconstrained joint model, ablation)
    'hard' : rule-violating states get probability exactly 0 (158 of 640 valid)
    'soft' : violating states are down-weighted by a learned log-penalty

Loss: negative log-likelihood of the observed (grade, lesions), summed over all
states compatible with the observed labels. Masked labels (e.g. the uncertain
IH entry of tr000215) are marginalized out instead of being dropped.
In 'hard' mode, training images whose labels break a rule have zero likelihood;
they are excluded from the joint NLL and counted (see `n_skipped`).

Optional auxiliary losses on the marginals (ASL on lesions, CE on grade) keep
the comparison with the ASL baseline fair; their weights are arguments.

Lesion order assumed: MA, HE, IH, VB/IRMA, NV, VH, RD  (indices 0..6).
Check this matches mmrdr_dataset.py before training.
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

N_GRADES = 5
N_LESIONS = 7
ADV_IDX = (4, 5, 6)  # NV, VH, RD


def _build_states():
    """All 640 joint states and their fixed feature matrix Phi."""
    grades, lesions = [], []
    for g in range(N_GRADES):
        for bits in itertools.product((0, 1), repeat=N_LESIONS):
            grades.append(g)
            lesions.append(bits)
    G = F.one_hot(torch.tensor(grades), N_GRADES).float()          # (640, 5)
    L = torch.tensor(lesions, dtype=torch.float32)                   # (640, 7)
    GL = (G.unsqueeze(2) * L.unsqueeze(1)).reshape(len(grades), -1)  # (640, 35)
    pairs = list(itertools.combinations(range(N_LESIONS), 2))
    LL = torch.stack([L[:, a] * L[:, b] for a, b in pairs], dim=1)   # (640, 21)
    Phi = torch.cat([G, L, GL, LL], dim=1)                           # (640, 68)
    return torch.tensor(grades), G, L, Phi


def _valid_mask(grades, L):
    any_lesion = L.sum(1) > 0
    any_adv = L[:, list(ADV_IDX)].sum(1) > 0
    r1 = (grades == 0) == (~any_lesion)
    r2 = (grades == 4) == any_adv
    return r1 & r2


class JointGradeLesionHead(nn.Module):
    def __init__(self, in_dim=512, mode="soft", init_violation_logit=-4.0,
                 learn_violation_logit=True):
        super().__init__()
        assert mode in ("none", "hard", "soft")
        self.mode = mode
        grades, G, L, Phi = _build_states()
        self.register_buffer("state_grade", grades)
        self.register_buffer("G", G)
        self.register_buffer("L", L)
        self.register_buffer("Phi", Phi)
        self.register_buffer("valid", _valid_mask(grades, L))
        self.proj = nn.Linear(in_dim, Phi.shape[1])
        # start close to an independent-heads model: interaction terms at zero
        with torch.no_grad():
            self.proj.weight[N_GRADES + N_LESIONS:].zero_()
            self.proj.bias[N_GRADES + N_LESIONS:].zero_()
        v = torch.tensor(float(init_violation_logit))
        if learn_violation_logit:
            self.violation_logit = nn.Parameter(v)
        else:
            self.register_buffer("violation_logit", v)

    def log_joint(self, h):
        """Log-probabilities over the 640 states, shape (B, 640). Always fp32."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = self.proj(h.float())
            E = z @ self.Phi.t()
            if self.mode == "hard":
                E = E.masked_fill(~self.valid, float("-inf"))
            elif self.mode == "soft":
                E = E + (~self.valid).float() * self.violation_logit
            return F.log_softmax(E, dim=1)

    def forward(self, h):
        """Returns grade marginals (B,5), lesion marginals (B,7), log joint (B,640)."""
        logp = self.log_joint(h)
        p = logp.exp()
        p_grade = p @ self.G
        p_lesion = (p @ self.L).clamp(1e-7, 1 - 1e-7)
        return p_grade, p_lesion, logp

    def nll(self, logp, y_grade, y_lesion, lesion_mask=None, reduction="mean"):
        """
        y_grade: (B,) long; y_lesion: (B,7) {0,1}; lesion_mask: (B,7) {0,1},
        1 = label observed, 0 = marginalize. Returns (loss, n_skipped);
        reduction 'mean' averages over usable rows, 'sum' sums them.
        """
        if lesion_mask is None:
            lesion_mask = torch.ones_like(y_lesion)
        obs = lesion_mask.bool().unsqueeze(1)                       # (B,1,7)
        match = (self.L.unsqueeze(0) == y_lesion.float().unsqueeze(1)) | ~obs
        compat = match.all(2) & (self.state_grade.unsqueeze(0) == y_grade.unsqueeze(1))
        # rows with no reachable compatible state (labels that break a rule in
        # 'hard' mode) are dropped BEFORE logsumexp: an all -inf row would give
        # NaN gradients even if it were filtered out afterwards
        ok = (compat & torch.isfinite(logp)).any(1)
        n_skipped = int((~ok).sum())
        if ok.sum() == 0:
            # zero that stays in the graph; logp may hold -inf, and -inf * 0 is NaN
            return logp.masked_fill(~torch.isfinite(logp), 0.0).sum() * 0.0, n_skipped
        masked = logp[ok].masked_fill(~compat[ok], float("-inf"))
        ll = torch.logsumexp(masked, dim=1)
        return (-ll.sum() if reduction == "sum" else -ll.mean()), n_skipped


def asl_on_probs(p, y, mask=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05,
                 reduction="mean"):
    """Asymmetric loss on probabilities (same settings as the paper's baseline).
    reduction 'none' returns the (B,7) elementwise loss (mask already applied)."""
    y = y.float()
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    with torch.no_grad():
        w_pos = (1 - p).pow(gamma_pos)
        w_neg = p_neg.pow(gamma_neg)
    loss = -(y * w_pos * torch.log(p.clamp(min=1e-8))
             + (1 - y) * w_neg * torch.log((1 - p_neg).clamp(min=1e-8)))
    if mask is not None:
        loss = loss * mask
    if reduction == "none":
        return loss
    if mask is not None:
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def joint_loss(head, h, y_grade, y_lesion, lesion_mask=None,
               w_nll=1.0, w_asl=1.0, w_ce=0.0):
    """Total loss: joint NLL + optional ASL on lesion marginals + optional grade CE."""
    p_grade, p_lesion, logp = head(h)
    nll, n_skipped = head.nll(logp, y_grade, y_lesion, lesion_mask)
    loss = w_nll * nll
    if w_asl > 0:
        loss = loss + w_asl * asl_on_probs(p_lesion, y_lesion, lesion_mask)
    if w_ce > 0:
        loss = loss + w_ce * F.nll_loss(torch.log(p_grade.clamp(min=1e-8)), y_grade)
    return loss, {"nll": nll.item(), "n_skipped": n_skipped,
                  "p_grade": p_grade, "p_lesion": p_lesion}


if __name__ == "__main__":
    torch.manual_seed(0)
    head = JointGradeLesionHead(512, mode="hard")
    print("states:", head.Phi.shape[0], "valid:", int(head.valid.sum()))
    assert head.Phi.shape == (640, 68)
    assert int(head.valid.sum()) == 158

    h = torch.randn(6, 512, requires_grad=True)
    y_g = torch.tensor([0, 1, 2, 3, 4, 4])
    y_l = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])  # last row breaks R1/R2
    mask = torch.ones_like(y_l)
    mask[2, 2] = 0  # uncertain IH, marginalized

    for mode in ("none", "hard", "soft"):
        head = JointGradeLesionHead(512, mode=mode)
        loss, info = joint_loss(head, h, y_g, y_l, mask)
        loss.backward()
        pg, pl = info["p_grade"], info["p_lesion"]
        assert torch.allclose(pg.sum(1), torch.ones(6), atol=1e-5)
        assert torch.isfinite(loss)
        assert h.grad is not None and torch.isfinite(h.grad).all()
        h.grad = None
        print(f"{mode:5s} loss={loss.item():.4f} skipped={info['n_skipped']}")

    # hard mode: a batch whose labels all break a rule must give a finite zero loss
    head = JointGradeLesionHead(512, mode="hard")
    hb = torch.randn(2, 512, requires_grad=True)
    _, _, lp = head(hb)
    bad_l = torch.zeros(2, 7, dtype=torch.long)
    z, ns = head.nll(lp, torch.tensor([1, 4]), bad_l, reduction="sum")
    z.backward()
    assert ns == 2 and z.item() == 0.0 and torch.isfinite(hb.grad).all()

    # hard mode: invalid states must have zero mass
    head = JointGradeLesionHead(512, mode="hard")
    _, _, logp = head(torch.randn(4, 512))
    assert torch.all(logp[:, ~head.valid] == float("-inf"))
    print("self-test passed")
