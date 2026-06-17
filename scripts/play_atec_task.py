# Created by skywoodsz on 2026/02/07.

import argparse
import math
import os
import time
import json

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from demo.solution import AlgSolution
solution = AlgSolution()

def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # TODO: simulate getting action spec from jason string (e.g. from a file or network)
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    action_spec_json = json.dumps(action_spec)

    # New Feature: apply safe action spec to env config (e.g. for scaling/clipping actions from your solution)
    env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video:
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Debug diagnostics setup
    # -------------------------------------------------------------------------
    if args_cli.debug:
        try:
            from isaaclab.sensors import ContactSensor
            robot = env.unwrapped.scene["robot"]

            # Try both sensor names: "contact_forces" (training) and "contact_sensor" (eval tasks)
            contact_sensor = env.unwrapped.scene.sensors.get("contact_forces", None)
            if contact_sensor is None:
                contact_sensor = env.unwrapped.scene.sensors.get("contact_sensor", None)

            # Get foot body names from robot config
            if hasattr(robot.cfg, "leg_joint_names"):
                foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            else:
                foot_names = None
                contact_sensor = None

            # Initialize duty factor tracking (2 seconds = 100 steps at 50Hz)
            duty_cycle_window = 100
            foot_contact_history = {name: [] for name in (foot_names or [])}

        except Exception as e:
            print(f"[DEBUG] Could not initialize contact sensor: {e}")
            contact_sensor = None
            foot_names = None
            foot_contact_history = {}

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            try:
                if isinstance(obs, dict):
                    robot_for_solution = env.unwrapped.scene["robot"]
                    root_pos_w = robot_for_solution.data.root_pos_w.detach()
                    root_quat_w = robot_for_solution.data.root_quat_w.detach()
                    qw = root_quat_w[:, 0]
                    qx = root_quat_w[:, 1]
                    qy = root_quat_w[:, 2]
                    qz = root_quat_w[:, 3]
                    root_yaw_w = torch.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )
                    obs["_root_pos_w"] = root_pos_w
                    obs["_root_yaw_w"] = root_yaw_w
            except Exception:
                pass

            resp = solution.predicts(obs, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            if not is_task_e and not args_cli.headless:
                camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                # Basic metrics
                print(f"\n[Step {timestep}] score={total_episode_reward:.2f}, elapsed={total_elapsed_time:.2f}s")

                # Robot state diagnostics
                try:
                    robot = env.unwrapped.scene["robot"]
                    root_pos = robot.data.root_pos_w[0].cpu().numpy()
                    root_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
                    root_ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
                    root_quat = robot.data.root_quat_w[0].cpu().numpy()
                    qw, qx, qy, qz = root_quat
                    root_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    print(f"  Root: pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}], "
                          f"vel_b=[{root_vel[0]:.2f}, {root_vel[1]:.2f}, {root_vel[2]:.2f}], "
                          f"ang_vel_b=[{root_ang_vel[0]:.2f}, {root_ang_vel[1]:.2f}, {root_ang_vel[2]:.2f}], "
                          f"yaw={root_yaw:+.2f}rad")

                    # Foot contact diagnostics with full details
                    if contact_sensor is not None and foot_names is not None:
                        try:
                            foot_ids = [contact_sensor.find_bodies([name])[0][0] for name in foot_names]
                            net_forces = contact_sensor.data.net_forces_w[0, foot_ids].cpu().numpy()
                            air_times = contact_sensor.data.current_air_time[0, foot_ids].cpu().numpy()
                            contact_times = contact_sensor.data.current_contact_time[0, foot_ids].cpu().numpy()

                            # Get foot positions in world and body frame
                            foot_pos_w = robot.data.body_pos_w[0, foot_ids].cpu().numpy()
                            foot_pos_b = torch.zeros(len(foot_ids), 3, device=robot.device)
                            for i in range(len(foot_ids)):
                                foot_pos_rel = foot_pos_w[i] - root_pos
                                # Transform to body frame using quaternion
                                from isaaclab.utils.math import quat_apply_inverse
                                quat_tensor = torch.tensor(root_quat, device=robot.device).unsqueeze(0)
                                pos_rel_tensor = torch.tensor(foot_pos_rel, device=robot.device).unsqueeze(0)
                                foot_pos_b[i] = quat_apply_inverse(quat_tensor, pos_rel_tensor)[0]
                            foot_pos_b = foot_pos_b.cpu().numpy()

                            print(f"  Feet:")
                            for i, name in enumerate(foot_names):
                                contact_force = float(torch.norm(torch.tensor(net_forces[i])).item())
                                is_contact = contact_force > 1.0
                                contact_status = "CONTACT" if is_contact else "AIR    "

                                # Update duty factor history
                                foot_contact_history[name].append(is_contact)
                                if len(foot_contact_history[name]) > duty_cycle_window:
                                    foot_contact_history[name].pop(0)

                                # Calculate duty factor (percentage of time in contact over last 2s)
                                if len(foot_contact_history[name]) > 0:
                                    duty_factor = sum(foot_contact_history[name]) / len(foot_contact_history[name])
                                else:
                                    duty_factor = 0.0

                                print(f"    {name:8s} (id={foot_ids[i]:2d}): {contact_status} "
                                      f"force={contact_force:6.1f}N air={air_times[i]:.3f}s contact={contact_times[i]:.3f}s "
                                      f"z={foot_pos_w[i][2]:.3f}m "
                                      f"pos_b=[{foot_pos_b[i][0]:+.3f},{foot_pos_b[i][1]:+.3f},{foot_pos_b[i][2]:+.3f}] "
                                      f"duty={duty_factor:.2%}")

                        except Exception as e:
                            print(f"  [Foot diagnostics error: {e}]")

                    # Leg action diagnostics (first 12 actions are legs for B2Piper)
                    leg_actions = actions[0, :12].cpu().numpy()
                    print(f"  Leg actions: FR=[{leg_actions[0]:+.2f},{leg_actions[1]:+.2f},{leg_actions[2]:+.2f}] "
                          f"FL=[{leg_actions[3]:+.2f},{leg_actions[4]:+.2f},{leg_actions[5]:+.2f}] "
                          f"RR=[{leg_actions[6]:+.2f},{leg_actions[7]:+.2f},{leg_actions[8]:+.2f}] "
                          f"RL=[{leg_actions[9]:+.2f},{leg_actions[10]:+.2f},{leg_actions[11]:+.2f}]")

                    # Arm actions if present
                    if actions.shape[1] >= 20:
                        arm_actions = actions[0, 12:20].cpu().numpy()
                        print(f"  Arm actions: [{arm_actions[0]:+.2f},{arm_actions[1]:+.2f},"
                              f"{arm_actions[2]:+.2f},{arm_actions[3]:+.2f},"
                              f"{arm_actions[4]:+.2f},{arm_actions[5]:+.2f},"
                              f"{arm_actions[6]:+.2f},{arm_actions[7]:+.2f}]")

                except Exception as e:
                    print(f"  [Diagnostics error: {e}]")

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
