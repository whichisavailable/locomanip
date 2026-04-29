# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.rough_env_cfg import (
    UnitreeGo2ArmRoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry import go2arm


def _resolve_init_noise_std(*, allow_vector: bool) -> float | tuple[float, ...]:
    """Read the rough-config init std and optionally collapse vector settings for legacy scalar-only policies."""
    init_noise_std = getattr(UnitreeGo2ArmRoughEnvCfg, "rsl_rl_init_noise_std", None)
    if init_noise_std is None:
        init_noise_std = getattr(UnitreeGo2ArmRoughEnvCfg(), "rsl_rl_init_noise_std", None)
    if init_noise_std is None:
        raise AttributeError("UnitreeGo2ArmRoughEnvCfg.rsl_rl_init_noise_std is not defined.")
    if allow_vector or isinstance(init_noise_std, float):
        return init_noise_std
    return float(sum(init_noise_std) / len(init_noise_std))


@configclass
class UnitreeGo2ArmRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "unitree_go2arm_rough"
    go2arm_mani_phase_reset_iteration = None
    go2arm_mani_phase_reset_arm_action_indices = tuple(range(12, 18))
    go2arm_mani_phase_reset_arm_std = 0.4
    go2arm_mani_phase_reset_learning_rate = 3.0e-4

    policy = RslRlPpoActorCriticCfg(
        # 先降低早期探索噪声，减少高噪声绝对位置动作带来的无效碰撞。
        init_noise_std=_resolve_init_noise_std(allow_vector=False),
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2ArmTeacherRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "unitree_go2arm_teacher_rough"
    go2arm_mani_phase_reset_iteration = None
    go2arm_mani_phase_reset_arm_action_indices = tuple(range(12, 18))
    go2arm_mani_phase_reset_arm_std = 0.4
    go2arm_mani_phase_reset_learning_rate = 3.0e-4

    # isaaclab_rl 0.4.x still supports remapping observation groups even though the policy
    # must be configured through the legacy single ActorCritic entry point.
    obs_groups = {
        "actor": ["policy", "privileged"],
        "critic": ["policy", "privileged", "critic_extra"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name=(
            '__import__("robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.'
            'legacy_actor_critic", fromlist=["PrivilegedTeacherActorCritic"]).PrivilegedTeacherActorCritic'
        ),
        # teacher 版本同样先降低早期探索噪声，优先观察姿态/接触问题而不是纯噪声扰动。
        init_noise_std=_resolve_init_noise_std(allow_vector=True),
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2ArmFlatPPORunnerCfg(UnitreeGo2ArmTeacherRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.experiment_name = "unitree_go2arm_teacher_flat"


@configclass
class UnitreeGo2ArmFlatPPORunnerWithSymmetryCfg(UnitreeGo2ArmFlatPPORunnerCfg):
    """Flat Go2Arm PPO config with world-XZ mirror data augmentation."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=go2arm.compute_symmetric_states,
        ),
    )
