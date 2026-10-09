import math
from contextlib import nullcontext
import torch
import torch.nn as nn

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    if not math.isfinite(reg) or reg <= 0:
        raise ValueError("Sinkhorn regularization must be finite and positive")
    if not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError("Sinkhorn requires at least one iteration")
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        # logsumexp already removes the reduced dimension. Retain the explicit
        # batch axis instead of relying on marginal broadcasting to restore it.
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
# 给原始的 cluster-feature score matrix S 增加一个 dustbin
# 然后用 Sinkhorn Optimal Transport，把原始 score 转换成一个满足特定行和列质量约束的 assignment matrix。
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """Log assignments with a scalar or per-patch [B, N] dustbin score.

    The prescribed dustbin marginal is N-M; a per-patch score redistributes
    this mass rather than specifying a new global rejection fraction.
    """

    batch_size, m, n = S.size()
    if n <= m:
        raise ValueError("SALAD requires more patches than clusters (N > M)")
    # 增加一个 dustbin row 到 score matrix S
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    # norm = -log(n + m)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    # log_a 是源权重，log_b 是目标权重
    # 源权重和目标权重的形状都是[B, num_clusters + 1]
    # log_a : [m+1], log_b : [n]
    log_a[-1] = log_a[-1] + math.log(n-m)
    # log_a[-1] = log_a[-1] + log(n-m)，调整 dustbin 的权重，使其与其他 cluster 的权重相匹配。
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    # log_a : [B_size, m+1], log_b : [B_size, n]
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    # log_P : [B_size, m+1, n+1]
    # log_P 是 Sinkhorn 算法的输出，形状为[B, num_clusters + 1, num_clusters + 1]
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
            reliability_ot=False,
            reliability_lambda=2.0,
            reliability_hidden_dim=64,
            reliability_context=False,
            reliability_detach_features=False,
            reliability_mode="learned",
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.reliability_ot = bool(reliability_ot)
        self.reliability_context_enabled = bool(reliability_context)
        self.reliability_detach_features = bool(reliability_detach_features)
        if reliability_mode not in ("learned", "fixed"):
            raise ValueError("reliability_mode must be 'learned' or 'fixed'")
        self.reliability_mode = reliability_mode
        self._reliability_lambda = float(reliability_lambda)
        if not math.isfinite(self._reliability_lambda) or self._reliability_lambda < 0:
            raise ValueError("reliability_lambda must be finite and nonnegative")
        if self.reliability_ot and (not isinstance(reliability_hidden_dim, int) or reliability_hidden_dim < 1):
            raise ValueError("reliability_hidden_dim must be a positive integer")

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # global token 的 Projection layer
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )

        # Dimension reduction for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )

        # Score projection for Sinkhorn algorithm
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))
        if self.reliability_ot:
            # Initialize after every original layer so an identical random seed
            # gives identical original SALAD weights for a fresh-mode ablation.
            # Persist strength for raw state_dict checkpoints as well as the
            # workflow's model_config. Original-mode keys remain unchanged.
            self.register_buffer("reliability_ot_lambda", torch.tensor(self._reliability_lambda))
            if self.reliability_mode == "fixed":
                self.register_buffer("reliability_fixed_mode", torch.tensor(True))
            # Isolate initialization from the training RNG. The legacy layers
            # above, and the very next random draw, agree with the off branch.
            # These modules are constructed on CPU, so no CUDA RNG is touched.
            with torch.random.fork_rng(devices=[]):
                self.reliability_head = nn.Sequential(
                    nn.Conv2d(self.num_channels, reliability_hidden_dim, 1),
                    nn.ReLU(),
                    nn.Conv2d(reliability_hidden_dim, 1, 1),
                )
                if self.reliability_context_enabled:
                    self.reliability_context = nn.Conv2d(
                        reliability_hidden_dim, reliability_hidden_dim, 3,
                        padding=1, groups=reliability_hidden_dim,
                    )
                # A constant row bias is a Sinkhorn gauge transformation. Thus
                # this initialization preserves pretrained assignments.
                nn.init.zeros_(self.reliability_head[-1].weight)
                nn.init.constant_(self.reliability_head[-1].bias, math.log(9.0))

    @property
    def reliability_lambda(self):
        if self.reliability_ot:
            return float(self.reliability_ot_lambda.detach().cpu())
        return self._reliability_lambda

    def reliability_parameters(self):
        """Parameters of the optional predictor, including its local context."""
        if self.reliability_ot:
            yield from self.reliability_head.parameters()
            if self.reliability_context_enabled:
                yield from self.reliability_context.parameters()

    def predict_reliability(self, features):
        """Predict FP32 logits; detach affects only the predictor input branch.

        The fixed mode retains checkpoint weights while omitting all predictor
        computation/gradients. It isolates the FP32 transport path in ablations.
        """
        if not self.reliability_ot:
            raise RuntimeError("predict_reliability requires reliability_ot=True")
        if features.ndim != 4 or features.shape[1] != self.num_channels:
            raise ValueError("reliability features must have shape [B,C,H,W]")
        if self.reliability_mode == "fixed":
            return torch.full((features.shape[0], 1, *features.shape[-2:]),
                              math.log(9.0), dtype=torch.float32, device=features.device)
        predictor_input = features.detach() if self.reliability_detach_features else features
        hidden = self.reliability_head[1](self.reliability_head[0](predictor_input))
        if self.reliability_context_enabled:
            hidden = hidden + nn.functional.relu(self.reliability_context(hidden))
        return self.reliability_head[2](hidden).float()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "reliability_ot_lambda"
        if self.reliability_ot and key in state_dict:
            strength = state_dict[key]
            if (not isinstance(strength, torch.Tensor) or strength.ndim != 0
                    or not torch.isfinite(strength).item() or strength.item() < 0):
                error_msgs.append(f"{key} must be a finite nonnegative scalar")
        fixed_key = prefix + "reliability_fixed_mode"
        if self.reliability_ot and self.reliability_mode == "fixed" and fixed_key in state_dict:
            fixed = state_dict[fixed_key]
            if (not isinstance(fixed, torch.Tensor) or fixed.ndim != 0
                    or fixed.dtype != torch.bool or not fixed.item()):
                error_msgs.append(f"{fixed_key} must be scalar boolean True")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


    def forward(self, x, return_aux=False):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x # Extract features and token
        # 右边的x是一个tuple，包含两个元素：f和t。f是局部特征张量，t是全局场景token张量。

        f = self.cluster_features(x).flatten(2)
        # flatten(2)将张量展平为二维张量，保留前两个维度不变，第三个维度被展平为一个维度。
        # 维度变化为[B, cluster_dim, H*W]，其中H和W是输入特征图的高度和宽度。
        p = self.score(x).flatten(2)
        t = self.token_features(t)# 维度变化为[B, token_dim]=[B,256]

        # Sinkhorn algorithm. The original branch deliberately retains its
        # operation order, dtype, parameters, and descriptor interface.
        reliability_logits = None
        if self.reliability_ot:
            reliability_logits = self.predict_reliability(x)
            reliability = torch.sigmoid(reliability_logits)
            dustbin_score = self.dust_bin.float() + self.reliability_ot_lambda.float() * (1 - reliability.flatten(1))
            precision_context = (
                torch.autocast(device_type=x.device.type, enabled=False)
                if x.device.type in ("cpu", "cuda") else nullcontext()
            )
            with precision_context:
                p = get_matching_probs(p.float(), dustbin_score, 3)
        else:
            p = get_matching_probs(p, self.dust_bin, 3)
        # p是匹配概率矩阵，形状为[B, num_clusters, num_clusters]。
        p = torch.exp(p)
        dustbin_probability = p[:, -1, :]
        # p 是 Sinkhorn 算法的输出，形状为[B, num_clusters, num_clusters]。
        p = p[:, :-1, :]
        keep_probability = p.sum(dim=1)
        # 去掉 dustbin 的概率，只保留 cluster 的概率。

        # 处理之前的p维度为[B, num_clusters, n]，f维度为[B, cluster_dim, n]。
        # num_clusters是聚类的数量，cluster_dim是每个聚类的特征维度，n是local features的数量。
        if self.reliability_ot:
            # Avoid a [B, cluster_dim, num_clusters, N] temporary for the new
            # branch. FP32 assignments and accumulation are AMP-safe.
            with precision_context:
                aggregated = torch.einsum("bcn,bkn->bck", f.float(), p)
                descriptor = torch.cat([
                    nn.functional.normalize(t.float(), p=2, dim=-1),
                    nn.functional.normalize(aggregated, p=2, dim=1).flatten(1),
                ], dim=-1)
                descriptor = nn.functional.normalize(descriptor, p=2, dim=-1)
            if return_aux:
                return descriptor, {
                    "reliability_logits": reliability_logits,
                    "reliability": reliability,
                    "local_features": x,
                    "assignment_dustbin": dustbin_probability,
                    "assignment_keep": keep_probability,
                    "dustbin_mass": dustbin_probability.sum(dim=1),
                }
            return descriptor

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        # p的维度变为[B, cluster_dim, num_clusters, n]，与f的维度对齐。
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        # f的维度变为[B, cluster_dim, num_clusters, H*W]，与p的维度对齐。

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            # 对第二个维度进行L2归一化，得到全局token的特征向量。
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
            # 先在最后一维度(dim=-1)进行加权求和，得到每个聚类的特征向量
            # 然后在第二个维度(dim=1)进行L2归一化
            # 最后在第一个维度(dim=1)进行展平，得到形状为[B, m*l]的特征向量
        ], dim=-1)
        # 在最后一个维度(dim=-1)进行拼接，得到形状为[B, m*l + g]的全局描述符f。

        descriptor = nn.functional.normalize(f, p=2, dim=-1)
        if return_aux:
            return descriptor, {
                "local_features": x,
                "assignment_dustbin": dustbin_probability,
                "assignment_keep": keep_probability,
                "dustbin_mass": dustbin_probability.sum(dim=1),
            }
        return descriptor
