# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from torch.distributions import Normal

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = object


class MaskedActionPPO(PPO):
    """PPO variant that ignores selected action dimensions in actor loss terms.

    The environment still receives the full action vector. The mask only changes policy log-probability, entropy, KL,
    and mirror-loss terms while it is active.
    """

    def __init__(
        self,
        *args,
        action_mask: list[bool] | tuple[bool, ...] | torch.Tensor | None = None,
        action_mask_until_iteration: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._action_mask_input = action_mask
        self.action_mask = self._resolve_action_mask_from_storage()
        self.action_mask_until_iteration = action_mask_until_iteration
        self._action_mask_update_count = 0
        self._debug_go2arm_mask = os.getenv("ROBOT_LAB_DEBUG_GO2ARM_MASK", "").lower() in ("1", "true", "yes")
        self._debug_go2arm_mask_interval = max(int(os.getenv("ROBOT_LAB_DEBUG_GO2ARM_MASK_INTERVAL", "50")), 1)
        self._debug_go2arm_structure_printed = False

    def init_storage(self, *args, **kwargs):
        result = super().init_storage(*args, **kwargs)
        self.action_mask = self._resolve_action_mask_from_storage(args[-1] if args else None)
        return result

    def act(self, *args, **kwargs) -> torch.Tensor:
        """Sample actions and store transition data with masked actor log-probability."""
        if self._uses_legacy_policy_api():
            return self._legacy_act(*args, **kwargs)

        obs = args[0]
        action_mask = self._active_action_mask()
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self._actor_log_prob(self.transition.actions, action_mask).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore[return-value]

    def update(self) -> dict[str, float]:
        """Run PPO updates using the active action-dimension mask."""
        if self._uses_legacy_policy_api():
            return self._legacy_update()

        action_mask = self._active_action_mask()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore[union-attr]

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self._actor_log_prob(batch.actions, action_mask)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self._actor_entropy(action_mask)[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self._actor_kl(batch.old_distribution_params, distribution_params, action_mask)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_prediction = mean_actions[original_batch_size:]
                symmetry_target = actions_mean_symm.detach()[original_batch_size:]
                if action_mask is not None:
                    symmetry_prediction = symmetry_prediction[..., action_mask]
                    symmetry_target = symmetry_target[..., action_mask]
                symmetry_loss = mse_loss(symmetry_prediction, symmetry_target)

                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore[index]
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._action_mask_update_count += 1

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        try:
            saved_dict = super().save()
        except AttributeError:
            saved_dict = {}
        saved_dict["action_mask_update_count"] = self._action_mask_update_count
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load models and restore mask timing."""
        try:
            load_iteration = super().load(loaded_dict, load_cfg, strict)
        except TypeError:
            load_iteration = super().load(loaded_dict)
        if load_iteration:
            self._action_mask_update_count = int(
                loaded_dict.get("action_mask_update_count", loaded_dict.get("iter", self._action_mask_update_count))
            )
        return load_iteration

    def _legacy_act(self, obs, critic_obs=None) -> torch.Tensor:
        """Legacy rsl_rl 2.x/3.x rollout step with masked actor log-probability."""
        if critic_obs is None:
            critic_obs = obs
        critic_obs = self._resolve_legacy_critic_obs(obs, critic_obs)

        action_mask = self._active_action_mask()
        policy = self._legacy_policy()
        self._debug_print_go2arm_structure(policy)
        if policy.is_recurrent:
            get_hidden_states = getattr(policy, "get_hidden_states", getattr(policy, "get_hidden_state", None))
            self.transition.hidden_states = get_hidden_states()

        self.transition.actions = policy.act(obs).detach()
        self.transition.values = policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self._legacy_policy_log_prob(policy, self.transition.actions, action_mask)
        self.transition.actions_log_prob = self.transition.actions_log_prob.detach()
        self.transition.action_mean = policy.action_mean.detach()
        self.transition.action_sigma = policy.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def _legacy_update(self):
        """Legacy rsl_rl PPO update with action-dimension masked actor terms."""
        action_mask = self._active_action_mask()
        policy = self._legacy_policy()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = self._parse_legacy_batch(batch)
            original_batch_size = obs_batch.shape[0]
            critic_obs_batch = self._resolve_legacy_critic_obs(obs_batch, critic_obs_batch)

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = self._legacy_augment(
                    data_augmentation_func, obs_batch, actions_batch, obs_type="policy"
                )
                critic_obs_batch, _ = self._legacy_augment(
                    data_augmentation_func, critic_obs_batch, None, obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self._legacy_policy_log_prob(policy, actions_batch, action_mask)
            value_batch = policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = policy.action_mean[:original_batch_size]
            sigma_batch = policy.action_std[:original_batch_size]
            entropy_batch = self._legacy_policy_entropy(policy, action_mask)[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self._legacy_policy_kl(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch, action_mask)
                    kl_mean = torch.mean(kl)

                    if getattr(self, "is_multi_gpu", False):
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if getattr(self, "gpu_global_rank", 0) == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if getattr(self, "is_multi_gpu", False):
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                if not self.symmetry["use_data_augmentation"]:
                    obs_batch, _ = self._legacy_augment(
                        data_augmentation_func, obs_batch, None, obs_type="policy"
                    )

                mean_actions_batch = policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = self._legacy_augment(
                    data_augmentation_func, None, action_mean_orig, obs_type="policy"
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_prediction = mean_actions_batch[original_batch_size:]
                symmetry_target = actions_mean_symm_batch.detach()[original_batch_size:]
                if action_mask is not None:
                    symmetry_prediction = symmetry_prediction[..., action_mask]
                    symmetry_target = symmetry_target[..., action_mask]
                symmetry_loss = mse_loss(symmetry_prediction, symmetry_target)

                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            self._debug_print_go2arm_gradients(policy)
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if getattr(self, "is_multi_gpu", False):
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._action_mask_update_count += 1

        if hasattr(self, "policy"):
            loss_dict = {
                "value_function": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "entropy": mean_entropy,
            }
            if self.rnd:
                loss_dict["rnd"] = mean_rnd_loss
            if self.symmetry:
                loss_dict["symmetry"] = mean_symmetry_loss
            return loss_dict
        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss

    def _resolve_action_mask_from_storage(self, fallback_action_shape=None) -> torch.Tensor | None:
        action_dim = None
        storage_actions = getattr(getattr(self, "storage", None), "actions", None)
        if storage_actions is not None:
            action_dim = int(storage_actions.shape[-1])
        elif fallback_action_shape is not None:
            if isinstance(fallback_action_shape, int):
                action_dim = fallback_action_shape
            elif isinstance(fallback_action_shape, (list, tuple)):
                action_dim = int(fallback_action_shape[-1])
        if action_dim is None:
            return None
        return self._resolve_action_mask(self._action_mask_input, action_dim)

    def _uses_legacy_policy_api(self) -> bool:
        return hasattr(self, "policy") or hasattr(self, "actor_critic")

    def _legacy_policy(self):
        policy = getattr(self, "policy", None)
        if policy is None:
            policy = getattr(self, "actor_critic", None)
        if policy is None:
            raise RuntimeError("MaskedActionPPO could not find a legacy policy module.")
        return policy

    def _debug_print_go2arm_structure(self, policy) -> None:
        if not self._debug_go2arm_mask or self._debug_go2arm_structure_printed:
            return
        self._debug_go2arm_structure_printed = True
        print("[go2arm debug] algorithm:", type(self).__name__)
        print("[go2arm debug] policy:", type(policy).__name__)
        print("[go2arm debug] actor:", type(getattr(policy, "actor", None)).__name__)
        print("[go2arm debug] action_mask:", self.action_mask)
        print("[go2arm debug] action_mask_until_iteration:", self.action_mask_until_iteration)

    def _debug_print_go2arm_gradients(self, policy) -> None:
        if not self._debug_go2arm_mask:
            return
        actor = getattr(policy, "actor", None)
        if actor is None or not hasattr(actor, "leg_head") or not hasattr(actor, "arm_head"):
            return
        update_count = int(self._action_mask_update_count)
        important_updates = {
            0,
            1,
            2,
            int(self.action_mask_until_iteration) if self.action_mask_until_iteration is not None else -1,
            int(self.action_mask_until_iteration) + 1 if self.action_mask_until_iteration is not None else -1,
        }
        if update_count not in important_updates and update_count % self._debug_go2arm_mask_interval != 0:
            return

        leg_grad = self._debug_grad_abs_sum(actor.leg_head)
        arm_grad = self._debug_grad_abs_sum(actor.arm_head)
        print(
            "[go2arm debug]",
            f"update={update_count}",
            f"mask_active={self._active_action_mask() is not None}",
            f"leg_head_grad={leg_grad:.6e}",
            f"arm_head_grad={arm_grad:.6e}",
        )

    @staticmethod
    def _debug_grad_abs_sum(module: nn.Module) -> float:
        total = 0.0
        for param in module.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().abs().sum().item())
        return total

    def _parse_legacy_batch(self, batch):
        if not isinstance(batch, (tuple, list)):
            raise TypeError(f"Unsupported legacy RSL-RL PPO batch type: {type(batch)!r}.")
        if len(batch) < 8:
            raise ValueError(
                f"Unsupported legacy RSL-RL PPO batch with {len(batch)} fields: {self._legacy_batch_shapes(batch)}."
            )

        action_dim = self._legacy_action_dim()
        obs_batch = batch[0]
        cursor = 1

        if cursor < len(batch) and self._is_legacy_obs_batch(batch[cursor]) and not self._is_action_batch(
            batch[cursor], action_dim
        ):
            critic_obs_batch = batch[cursor]
            cursor += 1
        else:
            critic_obs_batch = obs_batch

        if cursor >= len(batch) or not self._is_action_batch(batch[cursor], action_dim):
            action_cursor = None
            for index in range(cursor, len(batch)):
                if self._is_action_batch(batch[index], action_dim):
                    action_cursor = index
                    break
            if action_cursor is None:
                raise ValueError(
                    "Could not find the action tensor in legacy RSL-RL PPO batch. "
                    f"Expected last dimension {action_dim}; got {self._legacy_batch_shapes(batch)}."
                )
            if action_cursor > cursor:
                critic_obs_batch = batch[cursor]
            cursor = action_cursor

        actions_batch = batch[cursor]
        rest = batch[cursor + 1 :]
        if len(rest) < 6:
            raise ValueError(
                "Unsupported legacy RSL-RL PPO batch after action tensor. "
                f"Expected value/advantage/return/log-prob/mu/sigma fields; got {self._legacy_batch_shapes(batch)}."
            )

        target_values_batch = rest[0]
        advantages_batch = rest[1]
        returns_batch = rest[2]
        old_actions_log_prob_batch = rest[3]
        old_mu_batch = rest[4]
        old_sigma_batch = rest[5]
        remainder = rest[6:]

        hid_states_batch = (None, None)
        masks_batch = None
        rnd_state_batch = None
        if remainder:
            if len(remainder) >= 2 and isinstance(remainder[0], (tuple, list)):
                hid_states_batch = remainder[0]
                masks_batch = remainder[1]
                if len(remainder) >= 3:
                    rnd_state_batch = remainder[2]
            else:
                rnd_state_batch = remainder[0]

        return (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        )

    def _legacy_action_dim(self) -> int:
        if self.action_mask is not None:
            return int(self.action_mask.numel())
        if self._action_mask_input is not None:
            return int(torch.as_tensor(self._action_mask_input).numel())
        storage_actions = getattr(getattr(self, "storage", None), "actions", None)
        if storage_actions is not None:
            return int(storage_actions.shape[-1])
        return 18

    def _is_action_batch(self, value, action_dim: int) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[-1]) == action_dim

    def _is_legacy_obs_batch(self, value) -> bool:
        if isinstance(value, torch.Tensor):
            return value.ndim >= 2 and int(value.shape[-1]) in (92, 148, 149)
        return hasattr(value, "keys")

    def _legacy_batch_shapes(self, batch) -> tuple[str, ...]:
        shapes = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                shapes.append(str(tuple(item.shape)))
            elif isinstance(item, (tuple, list)):
                shapes.append(type(item).__name__)
            elif item is None:
                shapes.append("None")
            else:
                shapes.append(type(item).__name__)
        return tuple(shapes)

    def _resolve_legacy_critic_obs(self, obs, critic_obs):
        if isinstance(obs, torch.Tensor) and isinstance(critic_obs, torch.Tensor) and obs.shape[-1] == 148:
            if critic_obs.shape[-1] == 1:
                return torch.cat([obs, critic_obs], dim=-1)
            if critic_obs.shape[-1] == 148:
                critic_extra = torch.zeros(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)
                return torch.cat([obs, critic_extra], dim=-1)
        return critic_obs

    def _legacy_policy_log_prob(
        self, policy, actions: torch.Tensor, action_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if action_mask is None:
            return policy.get_actions_log_prob(actions)
        distribution = getattr(policy, "distribution", None)
        if distribution is None:
            raise RuntimeError("MaskedActionPPO requires policy.distribution for masked legacy log-probability.")
        return distribution.log_prob(actions)[..., action_mask].sum(dim=-1)

    def _legacy_policy_entropy(self, policy, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return policy.entropy
        distribution = getattr(policy, "distribution", None)
        if distribution is None:
            raise RuntimeError("MaskedActionPPO requires policy.distribution for masked legacy entropy.")
        return distribution.entropy()[..., action_mask].sum(dim=-1)

    def _legacy_policy_kl(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kl = torch.log(new_std / old_std + 1.0e-5) + (
            torch.square(old_std) + torch.square(old_mean - new_mean)
        ) / (2.0 * torch.square(new_std)) - 0.5
        if action_mask is not None:
            kl = kl[..., action_mask]
        return torch.sum(kl, dim=-1)

    def _legacy_augment(self, data_augmentation_func, obs, actions, *, obs_type: str):
        try:
            return data_augmentation_func(obs=obs, actions=actions, env=self.symmetry["_env"], obs_type=obs_type)
        except TypeError:
            return data_augmentation_func(
                obs=obs,
                actions=actions,
                env=self.symmetry["_env"],
                is_critic=(obs_type == "critic"),
            )

    def _resolve_action_mask(
        self,
        action_mask: list[bool] | tuple[bool, ...] | torch.Tensor | None,
        action_dim: int,
    ) -> torch.Tensor | None:
        if action_mask is None:
            return None
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if mask.ndim != 1 or mask.numel() != action_dim:
            raise ValueError(f"action_mask must be a 1D mask with {action_dim} entries, got shape {tuple(mask.shape)}.")
        if not torch.any(mask):
            raise ValueError("action_mask must keep at least one action dimension.")
        if torch.all(mask):
            return None
        return mask

    def _active_action_mask(self) -> torch.Tensor | None:
        if self.action_mask is None:
            return None
        if (
            self.action_mask_until_iteration is not None
            and self._action_mask_update_count >= self.action_mask_until_iteration
        ):
            return None
        return self.action_mask

    def _actor_log_prob(self, actions: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return self.actor.get_output_log_prob(actions)  # type: ignore[no-any-return]
        distribution = self._torch_distribution()
        return distribution.log_prob(actions)[..., action_mask].sum(dim=-1)

    def _actor_entropy(self, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return self.actor.output_entropy
        distribution = self._torch_distribution()
        return distribution.entropy()[..., action_mask].sum(dim=-1)

    def _actor_kl(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_mask is None:
            return self.actor.get_kl_divergence(old_params, new_params)  # type: ignore[no-any-return]
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist)[..., action_mask].sum(dim=-1)

    def _torch_distribution(self):
        distribution_module = getattr(self.actor, "distribution", None)
        distribution = getattr(distribution_module, "_distribution", None)
        if distribution is None:
            raise RuntimeError("MaskedActionPPO requires the actor distribution to be updated before masked terms.")
        return distribution
