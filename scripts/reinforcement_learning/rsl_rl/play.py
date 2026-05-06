# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--go2arm_ee_pos",
    type=float,
    nargs=3,
    metavar=("X_B", "Y_B", "Z_W"),
    default=None,
    help="Fixed Go2Arm ee target position for play: base-frame x/y and world-frame z.",
)
parser.add_argument(
    "--go2arm_ee_rpy",
    type=float,
    nargs=3,
    metavar=("ROLL_B", "PITCH_B", "YAW_B"),
    default=None,
    help="Fixed Go2Arm ee target orientation for play in base-frame roll/pitch/yaw radians.",
)
parser.add_argument(
    "--go2arm_trace_actions",
    action="store_true",
    default=False,
    help="Print Go2Arm action and joint state diagnostics during play.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _print_go2arm_termination_reasons(extras: dict) -> None:
    """Print per-env Go2Arm termination causes when the env exposes them."""
    termination_reasons = extras.get("go2arm_termination_reasons")
    if not termination_reasons:
        return

    for item in termination_reasons:
        env_id = item.get("env_id", "?")
        state = "truncated" if item.get("truncated") else "terminated"
        reasons = ", ".join(item.get("reasons", ())) or "unknown"
        print(f"[TERMINATION env{env_id}] {state}: {reasons}")


def _print_go2arm_action_state(env, policy_action: torch.Tensor, step: int) -> None:
    """Print one compact Go2Arm action snapshot for play-time debugging."""
    robot = env.unwrapped.scene["robot"]
    action_manager = env.unwrapped.action_manager

    effective_action = getattr(env.unwrapped, "_go2arm_effective_action", None)
    current_action = action_manager.action[0].detach().cpu()
    policy_action = policy_action.detach().cpu()
    if effective_action is not None:
        effective_action = effective_action[0].detach().cpu()
    else:
        effective_action = current_action
    prev_action = action_manager.prev_action[0].detach().cpu()
    joint_pos = robot.data.joint_pos[0].detach().cpu()
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu()
    root_pos_w = robot.data.root_pos_w[0].detach().cpu()
    root_quat_w = robot.data.root_quat_w[0].detach().cpu()
    ee_idx = robot.body_names.index("link6")
    ee_pos_w = robot.data.body_pos_w[0, ee_idx].detach().cpu()
    ee_quat_w = robot.data.body_quat_w[0, ee_idx].detach().cpu()
    ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    ee_pos_b = ee_pos_b.detach().cpu()

    joint_delta = joint_pos - default_joint_pos
    arm_action = current_action[-6:]
    effective_arm_action = effective_action[-6:]
    policy_arm_action = policy_action[-6:]
    arm_joint_delta = joint_delta[-6:]
    arm_joint_pos = joint_pos[-6:]

    print(
        f"[GO2ARM PLAY step={step}] "
        f"policy_norm={float(policy_action.norm().item()):.3f} "
        f"exec_norm={float(current_action.norm().item()):.3f} "
        f"prev_norm={float(prev_action.norm().item()):.3f} "
        f"base_w={[round(float(x), 3) for x in root_pos_w.tolist()]} "
        f"ee_w={[round(float(x), 3) for x in ee_pos_w.tolist()]} "
        f"ee_b={[round(float(x), 3) for x in ee_pos_b.tolist()]} "
        f"policy_arm={[round(float(x), 3) for x in policy_arm_action.tolist()]} "
        f"exec_arm={[round(float(x), 3) for x in effective_arm_action.tolist()]} "
        f"masked_arm={[round(float(x), 3) for x in arm_action.tolist()]} "
        f"arm_joint_delta={[round(float(x), 3) for x in arm_joint_delta.tolist()]} "
        f"arm_joint_pos={[round(float(x), 3) for x in arm_joint_pos.tolist()]}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    if "go2arm" in task_name.lower():
        env_cfg.enable_play_termination_reason_logging = True
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
            # Play starts a fresh environment counter at 0, so the training-time fixed arm freeze
            # would otherwise be active again for many steps. Disable it here to observe the
            # policy's real arm output during play.
            env_cfg.actions.joint_pos.fixed_delta_action_until_iteration = 0
            print("[INFO] Go2Arm play override: disabled fixed arm-action freeze for playback.")
    go2arm_fixed_target = args_cli.go2arm_ee_pos is not None or args_cli.go2arm_ee_rpy is not None
    if go2arm_fixed_target and hasattr(env_cfg.curriculum, "go2arm_reaching_stages"):
        env_cfg.curriculum.go2arm_reaching_stages = None
        if hasattr(env_cfg.commands, "ee_pose"):
            if args_cli.go2arm_ee_pos is not None:
                ee_x_b, ee_y_b, ee_z_w = args_cli.go2arm_ee_pos
                env_cfg.commands.ee_pose.position_range_b = (ee_x_b, ee_x_b, ee_y_b, ee_y_b, 0.0, 0.0)
                env_cfg.commands.ee_pose.world_z_range = (ee_z_w, ee_z_w)
            else:
                env_cfg.commands.ee_pose.position_range_b = (0.05, 2.00, -0.35, 0.35, 0.0, 0.0)
                env_cfg.commands.ee_pose.world_z_range = (0.02, 1.20)
            if args_cli.go2arm_ee_rpy is not None:
                ee_roll_b, ee_pitch_b, ee_yaw_b = args_cli.go2arm_ee_rpy
                env_cfg.commands.ee_pose.euler_xyz_range_b = (
                    ee_roll_b,
                    ee_roll_b,
                    ee_pitch_b,
                    ee_pitch_b,
                    ee_yaw_b,
                    ee_yaw_b,
                )
            print(
                f"[INFO] Go2Arm fixed ee command override: pos={args_cli.go2arm_ee_pos}, rpy={args_cli.go2arm_ee_rpy}"
            )
            env_cfg.commands.ee_pose.sample_z_in_world_frame = True
            env_cfg.commands.ee_pose.reject_position_cuboid = None
            env_cfg.commands.ee_pose.max_sampling_tries = 1
            env_cfg.commands.ee_pose.secondary_position_range_b = None
            env_cfg.commands.ee_pose.secondary_euler_xyz_range_b = None
            env_cfg.commands.ee_pose.secondary_world_z_range = None
            env_cfg.commands.ee_pose.secondary_sample_prob = 0.0
            env_cfg.commands.ee_pose.tertiary_position_range_b = None
            env_cfg.commands.ee_pose.tertiary_euler_xyz_range_b = None
            env_cfg.commands.ee_pose.tertiary_world_z_range = None
            env_cfg.commands.ee_pose.tertiary_sample_prob = 0.0
        env_cfg.events.randomize_reset_joints.params["position_range"] = (-0.04, 0.04)
        env_cfg.events.randomize_reset_joints.params["velocity_range"] = (-0.05, 0.05)
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.06, 0.06),
            "y": (-0.06, 0.06),
            "yaw": (-0.18, 0.18),
        }

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 导出策略到 jit / onnx。
    # 只要策略对象自己提供了 as_jit()/as_onnx()，就直接使用策略自带的导出包装。
    # 这样可以确保 go2arm 这类自定义 privileged teacher policy 走正确的导出语义。
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if hasattr(policy_nn, "as_jit") and hasattr(policy_nn, "as_onnx"):
        os.makedirs(export_model_dir, exist_ok=True)

        # 直接导出 TorchScript。
        jit_model = policy_nn.as_jit()
        jit_model.to("cpu")
        torch.jit.script(jit_model).save(os.path.join(export_model_dir, "policy.pt"))

        # 直接导出 ONNX。
        onnx_model = policy_nn.as_onnx(verbose=False)
        onnx_model.to("cpu")
        onnx_model.eval()
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            os.path.join(export_model_dir, "policy.onnx"),
            export_params=True,
            opset_version=18,
            verbose=False,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
        )
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    use_mean_action = "go2arm" in task_name.lower()
    if use_mean_action and not hasattr(policy_nn, "act_inference"):
        print("[INFO] Go2Arm play fallback: policy has no act_inference(); using policy(obs) instead.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    trace_step = 0
    trace_interval = 20 if args_cli.go2arm_trace_actions else None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if use_mean_action and hasattr(policy_nn, "act_inference"):
                actions = policy_nn.act_inference(obs)
            else:
                actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            _print_go2arm_termination_reasons(extras)
            if trace_interval is not None and timestep % trace_interval == 0:
                _print_go2arm_action_state(env, actions[0], trace_step)
            trace_step += 1
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
