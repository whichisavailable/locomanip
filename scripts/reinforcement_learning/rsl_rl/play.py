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
import math
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
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _format_pitch_debug_line(env) -> str | None:
    """Return a compact pitch debug line for env 0."""
    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        quat_wxyz = robot.data.root_quat_w[0]
        gravity_b = robot.data.projected_gravity_b[0]

        w, x, y, z = quat_wxyz.unbind(dim=-1)
        reward_pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        gravity_pitch = torch.atan2(
            -gravity_b[0],
            torch.sqrt(gravity_b[1] * gravity_b[1] + gravity_b[2] * gravity_b[2]),
        )

        front_rear_dz_text = "n/a"
        body_names = getattr(robot, "body_names", [])
        front_names = ("FL_hip", "FR_hip")
        rear_names = ("RL_hip", "RR_hip")
        if all(name in body_names for name in (*front_names, *rear_names)):
            front_ids = [body_names.index(name) for name in front_names]
            rear_ids = [body_names.index(name) for name in rear_names]
            body_pos_w = robot.data.body_pos_w[0]
            front_z = torch.mean(body_pos_w[front_ids, 2])
            rear_z = torch.mean(body_pos_w[rear_ids, 2])
            front_rear_dz = front_z - rear_z
            front_rear_dz_text = f"{front_rear_dz.item():+.4f} m"

        return (
            "[PITCH env0] "
            f"reward_pitch={reward_pitch.item():+.4f} rad/{math.degrees(reward_pitch.item()):+.1f} deg, "
            f"gravity_pitch={gravity_pitch.item():+.4f} rad/{math.degrees(gravity_pitch.item()):+.1f} deg, "
            f"front_rear_dz={front_rear_dz_text}"
        )
    except Exception as exc:
        return f"[PITCH env0] unavailable: {exc}"


def _format_go2arm_tracking_debug_lines(env) -> list[str]:
    """Return Go2Arm play debug lines for env 0."""
    lines: list[str] = []
    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        quat_wxyz = robot.data.root_quat_w[0]
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        lines.append(f"[PITCH env0] {pitch.item():+.4f} rad / {math.degrees(pitch.item()):+.2f} deg")
    except Exception as exc:
        lines.append(f"[PITCH env0] unavailable: {exc}")

    try:
        command_term = env.unwrapped.command_manager.get_term("ee_pose")
        position_error = command_term.position_tracking_error[0]
        orientation_error = command_term.orientation_tracking_error[0]
        lines.append(f"[EE POSITION ERROR env0] {position_error.item():.6f}")
        lines.append(f"[EE ORIENTATION ERROR env0] {orientation_error.item():.6f}")
    except Exception as exc:
        lines.append(f"[EE POSITION ERROR env0] unavailable: {exc}")
        lines.append(f"[EE ORIENTATION ERROR env0] unavailable: {exc}")

    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        leg_names = ("FL", "FR", "RL", "RR")
        joint_names = [f"{leg}_{joint}_joint" for leg in leg_names for joint in ("hip", "thigh", "calf")]
        joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
        joint_delta = robot.data.joint_pos[0, joint_ids] - robot.data.default_joint_pos[0, joint_ids]
        grouped_delta = joint_delta.reshape(len(leg_names), 3)
        delta_text = " ".join(
            f"{leg}=({values[0].item():+.2f},{values[1].item():+.2f},{values[2].item():+.2f})"
            for leg, values in zip(leg_names, grouped_delta)
        )
        lines.append(f"[LEG JOINT DELTA env0] {delta_text}")

        foot_names = tuple(f"{leg}_foot" for leg in leg_names)
        foot_ids, _ = robot.find_bodies(foot_names, preserve_order=True)
        foot_z = robot.data.body_pos_w[0, foot_ids, 2]
        foot_z_text = " ".join(f"{leg}={value.item():+.3f}" for leg, value in zip(leg_names, foot_z))
        lines.append(f"[FOOT Z env0] {foot_z_text}")
        foot_pos_rel_w = robot.data.body_pos_w[0, foot_ids] - robot.data.root_pos_w[0].unsqueeze(0)
        foot_pos_b = quat_apply_inverse(robot.data.root_quat_w[0].unsqueeze(0), foot_pos_rel_w)
        foot_pos_b_text = " ".join(
            f"{leg}=({pos[0].item():+.3f},{pos[1].item():+.3f},{pos[2].item():+.3f})"
            for leg, pos in zip(leg_names, foot_pos_b)
        )
        lines.append(f"[FOOT POS B env0 xyz] {foot_pos_b_text}")

        force_values = []
        force_z_values = []
        matrix_values = []
        matrix_z_values = []
        for leg in leg_names:
            sensor = scene.sensors.get(f"{leg}_foot_contact")
            if sensor is None:
                force_values.append(float("nan"))
                force_z_values.append(float("nan"))
                matrix_values.append(float("nan"))
                matrix_z_values.append(float("nan"))
                continue
            net_force = sensor.data.net_forces_w[0, 0]
            force_values.append(torch.linalg.norm(net_force).item())
            force_z_values.append(net_force[2].item())
            if sensor.data.force_matrix_w is not None:
                matrix_force = torch.sum(sensor.data.force_matrix_w[0, 0], dim=0)
                matrix_values.append(torch.linalg.norm(matrix_force).item())
                matrix_z_values.append(matrix_force[2].item())
            else:
                matrix_values.append(float("nan"))
                matrix_z_values.append(float("nan"))
        force_text = " ".join(f"{leg}={value:.2f}" for leg, value in zip(leg_names, force_values))
        lines.append(f"[FOOT CONTACT FORCE env0] {force_text}")
        force_z_text = " ".join(f"{leg}={value:+.2f}" for leg, value in zip(leg_names, force_z_values))
        lines.append(f"[FOOT CONTACT FORCE Z env0] {force_z_text} sum={sum(force_z_values):+.2f}")
        matrix_text = " ".join(f"{leg}={value:.2f}" for leg, value in zip(leg_names, matrix_values))
        lines.append(f"[FOOT MATRIX FORCE env0] {matrix_text}")
        matrix_z_text = " ".join(f"{leg}={value:+.2f}" for leg, value in zip(leg_names, matrix_z_values))
        lines.append(f"[FOOT MATRIX FORCE Z env0] {matrix_z_text} sum={sum(matrix_z_values):+.2f}")

        global_sensor = scene.sensors.get("contact_forces")
        if global_sensor is not None:
            global_foot_ids, _ = global_sensor.find_bodies(foot_names, preserve_order=True)
            global_foot_forces = global_sensor.data.net_forces_w[0, global_foot_ids]
            global_force_norm = torch.linalg.norm(global_foot_forces, dim=-1)
            global_force_text = " ".join(
                f"{leg}={value.item():.2f}" for leg, value in zip(leg_names, global_force_norm)
            )
            lines.append(f"[GLOBAL FOOT FORCE env0] {global_force_text}")
            global_force_z = global_foot_forces[:, 2]
            global_force_z_text = " ".join(
                f"{leg}={value.item():+.2f}" for leg, value in zip(leg_names, global_force_z)
            )
            lines.append(f"[GLOBAL FOOT FORCE Z env0] {global_force_z_text} sum={torch.sum(global_force_z).item():+.2f}")
            all_body_force_norm = torch.linalg.norm(global_sensor.data.net_forces_w[0], dim=-1)
            top_count = min(8, all_body_force_norm.numel())
            top_values, top_ids = torch.topk(all_body_force_norm, k=top_count)
            top_body_text = " ".join(
                f"{global_sensor.body_names[int(body_id)]}={value.item():.2f}"
                for body_id, value in zip(top_ids, top_values)
            )
            non_foot_ids = [
                body_id for body_id, body_name in enumerate(global_sensor.body_names) if body_name not in foot_names
            ]
            if non_foot_ids:
                non_foot_forces = all_body_force_norm[torch.as_tensor(non_foot_ids, device=all_body_force_norm.device)]
                non_foot_top_value, non_foot_top_pos = torch.max(non_foot_forces, dim=0)
                non_foot_top_id = non_foot_ids[int(non_foot_top_pos.item())]
                lines.append(
                    "[GLOBAL TOP BODY FORCE env0] "
                    f"{top_body_text}; max_non_foot={global_sensor.body_names[non_foot_top_id]}:{non_foot_top_value.item():.2f}"
                )

        joint_vel = robot.data.joint_vel[0, joint_ids].reshape(len(leg_names), 3)
        vel_text = " ".join(
            f"{leg}=({values[0].item():+.2f},{values[1].item():+.2f},{values[2].item():+.2f})"
            for leg, values in zip(leg_names, joint_vel)
        )
        lines.append(f"[LEG JOINT VEL env0] {vel_text}")

        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        leg_action_ids = [action_term._joint_names.index(name) for name in joint_names]
        leg_raw_action = action_term.raw_actions[0, leg_action_ids].reshape(len(leg_names), 3)
        raw_action_text = " ".join(
            f"{leg}=({values[0].item():+.2f},{values[1].item():+.2f},{values[2].item():+.2f})"
            for leg, values in zip(leg_names, leg_raw_action)
        )
        lines.append(f"[LEG RAW ACTION env0] {raw_action_text}")
        leg_target_delta = (
            action_term.processed_actions[0, leg_action_ids] - robot.data.default_joint_pos[0, joint_ids]
        ).reshape(len(leg_names), 3)
        target_delta_text = " ".join(
            f"{leg}=({values[0].item():+.2f},{values[1].item():+.2f},{values[2].item():+.2f})"
            for leg, values in zip(leg_names, leg_target_delta)
        )
        lines.append(f"[LEG TARGET DELTA env0] {target_delta_text}")
    except Exception as exc:
        lines.append(f"[LEG/FOOT DEBUG env0] unavailable: {exc}")

    return lines


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
                "[INFO] Go2Arm fixed ee command override: "
                f"pos={args_cli.go2arm_ee_pos}, rpy={args_cli.go2arm_ee_rpy}"
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

    # reset environment
    obs = env.get_observations()
    timestep = 0
    last_time_print = time.time()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
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

        # print current time every 1 second
        now = time.time()
        if now - last_time_print >= 1.0:
            print(f"[TIME] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
            for debug_line in _format_go2arm_tracking_debug_lines(env):
                print(debug_line)
            last_time_print = now

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
