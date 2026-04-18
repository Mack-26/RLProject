from dataclasses import dataclass, field, MISSING
from typing import Dict, Iterable, List, Tuple, Type

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

ALL_FEATURES = ["global", "local", "disagree", "team_var", "risk", "ratio"]


class AdaptiveAlphaNetwork(nn.Module):
    """Learns a per-agent mixing weight α ∈ (0,1) to blend MAPPO and IPPO value estimates.

    Features can be ablated via the `features` argument — any subset of ALL_FEATURES.
    """

    def __init__(self, features: List[str] = None):
        super().__init__()
        self.features = features if features is not None else ALL_FEATURES

        # Scalar weights for each active feature
        for name in ALL_FEATURES:
            if name in self.features:
                self.register_parameter(f"w_{name}", nn.Parameter(torch.randn(1) * 0.01))

        self.phi = nn.Parameter(torch.zeros(1))

    def _get_w(self, name: str) -> torch.Tensor:
        return getattr(self, f"w_{name}")

    def forward(self, v_global: torch.Tensor, v_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.phi.clone()

        if "global" in self.features:
            logits = logits + self._get_w("global") * v_global
        if "local" in self.features:
            logits = logits + self._get_w("local") * v_local
        if "disagree" in self.features:
            logits = logits + self._get_w("disagree") * torch.abs(v_global - v_local)
        if "team_var" in self.features:
            if v_local.shape[-2] > 1:
                f_team_var = v_local.var(dim=-2, keepdim=True).expand_as(v_local)
            else:
                f_team_var = torch.zeros_like(v_local)
            logits = logits + self._get_w("team_var") * f_team_var
        if "risk" in self.features:
            logits = logits + self._get_w("risk") * torch.minimum(v_global, v_local)
        if "ratio" in self.features:
            logits = logits + self._get_w("ratio") * (v_global / (torch.abs(v_local) + 1e-5))

        alpha = torch.sigmoid(logits)
        v_mix = alpha * v_global + (1.0 - alpha) * v_local
        return v_mix, alpha


class AdaptiveMappo(Algorithm):
    """MAPPO + IPPO blended via a learned α from AdaptiveAlphaNetwork.

    v_mix = α(v_global, v_local) * v_MAPPO + (1 - α) * v_IPPO

    Args:
        share_param_critic: share critic parameters within agent groups.
        clip_epsilon: PPO clip threshold.
        entropy_coef: entropy bonus coefficient.
        critic_coef: critic loss coefficient.
        loss_critic_type: "l1", "l2", or "smooth_l1".
        lmbda: GAE lambda.
        scale_mapping: std mapping ("softplus", "exp", "relu", "biased_softplus_1").
        use_tanh_normal: use TanhNormal for continuous actions.
        minibatch_advantage: compute GAE on minibatches to save memory.
        features: subset of ["global","local","disagree","team_var","risk","ratio"].
    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        features: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage
        self.features = features if features is not None else ALL_FEATURES

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(self, group: str, policy_for_loss: TensorDictModule, continuous: bool) -> Tuple[LossModule, bool]:
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(self, group: str, model_config: ModelConfig, continuous: bool) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite({group: self.observation_spec[group].clone().to(self.device)})
        actor_output_spec = Composite(
            {group: Composite({"logits": Unbounded(shape=logits_shape)}, shape=(n_agents,))}
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=IndependentNormal if not self.use_tanh_normal else TanhNormal,
                distribution_kwargs=(
                    {"low": self.action_spec[(group, "action")].space.low,
                     "high": self.action_spec[(group, "action")].space.high}
                    if self.use_tanh_normal else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={"logits": (group, "logits"), "mask": (group, "action_mask")},
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        return policy

    def _get_policy_for_collection(self, policy_for_loss: TensorDictModule, group: str, continuous: bool) -> TensorDictModule:
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(nested_done_key, batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)))
        if nested_terminated_key not in keys:
            batch.set(nested_terminated_key, batch.get(("next", "terminated")).unsqueeze(-1).expand((*group_shape, 1)))
        if nested_reward_key not in keys:
            batch.set(nested_reward_key, batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)))

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1

        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minibatch = batch[last_start_index:start_index]
            minibatches.append(minibatch)
            with torch.no_grad():
                loss.value_estimator(
                    minibatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        return torch.cat(minibatches, dim=0)

    def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
        loss_vals.set("loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"])
        del loss_vals["loss_entropy"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        if self.share_param_critic:
            global_critic_output_spec = Composite({"state_value_global": Unbounded(shape=(1,))})
        else:
            global_critic_output_spec = Composite(
                {group: Composite({"state_value_global": Unbounded(shape=(n_agents, 1))}, shape=(n_agents,))}
            )

        local_critic_output_spec = Composite(
            {group: Composite({"state_value_local": Unbounded(shape=(n_agents, 1))}, shape=(n_agents,))}
        )

        if self.state_spec is not None:
            global_input_has_agent_dim = False
            global_critic_input_spec = self.state_spec
        else:
            global_input_has_agent_dim = True
            global_critic_input_spec = Composite({group: self.observation_spec[group].clone().to(self.device)})

        local_critic_input_spec = Composite({group: self.observation_spec[group].clone().to(self.device)})

        global_value_module = self.critic_model_config.get_model(
            input_spec=global_critic_input_spec,
            output_spec=global_critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=global_input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        if self.share_param_critic:
            global_expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(*value.shape[:-1], n_agents, 1),
                in_keys=["state_value_global"],
                out_keys=[(group, "state_value_global")],
            )
            global_value_module = TensorDictSequential(global_value_module, global_expand_module)

        local_value_module = self.critic_model_config.get_model(
            input_spec=local_critic_input_spec,
            output_spec=local_critic_output_spec,
            n_agents=n_agents,
            centralised=False,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        adaptive_net = AdaptiveAlphaNetwork(features=self.features).to(self.device)
        if not hasattr(self, "_adaptive_nets"):
            self._adaptive_nets = {}
        self._adaptive_nets[group] = adaptive_net

        mix_module = TensorDictModule(
            adaptive_net,
            in_keys=[(group, "state_value_global"), (group, "state_value_local")],
            out_keys=[(group, "state_value"), (group, "alpha_weight")],
        )

        return TensorDictSequential(global_value_module, local_value_module, mix_module)


@dataclass
class AdaptiveMappoConfig(AlgorithmConfig):
    """Configuration dataclass for AdaptiveMappo."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    features: List[str] = field(default_factory=lambda: list(ALL_FEATURES))

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return AdaptiveMappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
