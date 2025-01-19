""" Runner for ZBot """
import argparse
import functools
import logging
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax import numpy as jp
from ml_collections import config_dict
from zbot import joystick as zbot_joystick
from zbot import randomize as zbot_randomize
from zbot import zbot_constants



from mujoco_playground import wrapper
from mujoco_playground._src.gait import draw_joystick_command


class ZBotRunner:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        """Initialize the ZBotRunner class.

        Args:
            args (argparse.Namespace): Command line arguments.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.args = args
        self.env_name = args.env
        self.setup_environment()
        self.setup_training_config()
        self.x_data, self.y_data, self.y_dataerr = [], [], []
        self.times = [datetime.now()]
        self.base_body = "Z-BOT2_MASTER-BODY-SKELETON"
        
    def setup_environment(self) -> None:
        """Initialize environment configuration"""
        self.env_config = zbot_joystick.default_config()
        self.env = zbot_joystick.Joystick(task=self.args.task)
        self.eval_env = zbot_joystick.Joystick(task=self.args.task)
        self.randomizer = zbot_randomize.domain_randomize
        
    def setup_training_config(self) -> None:
        """Setup PPO training configuration"""
        self.rl_config = config_dict.create(
            num_timesteps=5 if self.args.debug else 150_000_000,
            num_evals=15,
            reward_scaling=1.0,
            episode_length=1 if self.args.debug else self.env_config.episode_length,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=1 if self.args.debug else 32,
            num_updates_per_batch=1 if self.args.debug else 4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=0.005,
            num_envs=1 if self.args.debug else 8192,
            batch_size=2 if self.args.debug else 256,
            max_grad_norm=1.0,
            clipping_epsilon=0.2,
            num_resets_per_eval=1
        )
        
        self.rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
        self.logger.info(f"RL config: {self.rl_config}")

    def save_video(self, frames: list[np.ndarray], fps: float, filename: str = 'output.mp4') -> None:
        """Save video frames using OpenCV
        
        Args:
            frames (list[np.ndarray]): List of frames to save.
            fps (float): Frames per second.
            filename (str): Output filename.
        """
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        """Callback function for training progress
        
        Args:
            num_steps (int): Number of steps taken.
            metrics (dict): Metrics from the training process.
        """
        plt.figure()
        self.times.append(datetime.now())
        self.x_data.append(num_steps)
        self.y_data.append(metrics["eval/episode_reward"])
        self.y_dataerr.append(metrics["eval/episode_reward_std"])
        plt.xlim([0, self.rl_config["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.y_data[-1]:.3f}")
        plt.errorbar(self.x_data, self.y_data, yerr=self.y_dataerr, color="blue")
        plt.savefig('plot.png')
        plt.close()

    def train(self) -> None:
        """Execute training process"""
        ppo_training_params = dict(self.rl_config)
        if "network_factory" in self.rl_config:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **self.rl_config.network_factory
            )
            del ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks

        train_fn = functools.partial(
            ppo.train, **ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback
        )

        self.make_inference_fn, self.params, metrics = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        self.logger.info(f"Time to jit: {self.times[1] - self.times[0]}")
        self.logger.info(f"Time to train: {self.times[-1] - self.times[1]}")
        
        if self.args.save_model:
            self.save_model()

    def save_model(self) -> None:
        """Save model parameters"""
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"{self.env_name}_params.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.params, f)
        self.logger.info(f"Model saved to: {model_path}")

    def load_model(self) -> None:
        """Load model parameters"""
        model_path = Path("checkpoints") / f"{self.env_name}_params.pkl"
        with open(model_path, "rb") as f:
            self.params = pickle.load(f)
        
        if "network_factory" in self.rl_config:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **self.rl_config.network_factory
            )
        else:
            network_factory = ppo_networks.make_ppo_networks
        
        key = jax.random.PRNGKey(0)
        eval_env = zbot_joystick.Joystick(task=self.args.task)
        jit_reset = jax.jit(eval_env.reset)

        state = jit_reset(key)
        obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)

        ppo_network = network_factory(
            obs_shape, self.env.action_size)

        self.make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        self.logger.info("Model loaded successfully")

    @functools.partial(jax.jit, static_argnums=(0,))
    def run_step(
        self, state: jax.Array, rng: jax.Array, 
        inference_fn: any
    ) -> tuple[jax.Array, jax.Array]:
        """Execute a single environment step with JAX transformation
        
        Args:
            state (jax.Array): Current state.
            rng (jax.Array): Random number generator.
            inference_fn (any): Inference function.
        """
        act_rng, next_rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        next_state = self.eval_env.step(state, ctrl)
        return next_state, next_rng

    def evaluate(self) -> None:
        """Run evaluation episodes"""
        eval_env = zbot_joystick.Joystick(task=self.args.task)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(self.make_inference_fn(self.params, deterministic=True))

        # inference_fn = self.make_inference_fn(self.params, deterministic=True)
        rng = jax.random.PRNGKey(self.args.seed)
        command = jp.array([self.args.x_vel, self.args.y_vel, self.args.yaw_vel])
        phase_dt = 2 * jp.pi * self.eval_env.dt * 1.5
        phase = jp.array([0, jp.pi])
        
        for episode in range(self.args.num_episodes):
            self.logger.info(f"Episode {episode}")
            rollout = []
            modify_scene_fns = []
            
            # Initialize episode
            state = jit_reset(jax.random.PRNGKey(1))
            state.info["phase_dt"] = phase_dt
            state.info["phase"] = phase
            state.info["command"] = command
            
            # Run episode
            for _ in range(self.args.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                if state.done:
                    break
                state.info["command"] = command
                rollout.append(state)

                # Get robot position and orientation
                xyz = np.array(state.data.xpos[self.eval_env.mj_model.body(zbot_constants.ROOT_BODY).id])
                xyz += np.array([0, 0.0, 0])
                x_axis = state.data.xmat[self.eval_env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=np.linalg.norm(state.info["command"]),
                    )
                )
            
            self.render_episode(rollout, modify_scene_fns, episode)

    def render_episode(
        self, rollout: list[jax.Array], modify_scene_fns: list[callable], episode_num: int
    ) -> None:
        """Render and save episode video
        
        Args:
            rollout (list[jax.Array]): Rollout data.
            modify_scene_fns (list[callable]): Modify scene functions.
            episode_num (int): Episode number.
        """
        render_every = 1
        fps = 1.0 / self.eval_env.dt / render_every
        self.logger.info(f"fps: {fps}")
        
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

        frames = self.eval_env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640*2,
            height=480,
            modify_scene_fns=mod_fns,
        )

        self.save_video(frames, fps=fps, filename=f'output_{episode_num}.mp4')


def main():
    parser = argparse.ArgumentParser(description='ZBot Runner Script')
    parser.add_argument('--env', type=str, default="ZbotJoystickFlatTerrain", help='Environment to run')
    parser.add_argument('--task', type=str, default="flat_terrain", help='Task to run')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal parameters')
    parser.add_argument('--save-model', action='store_true', help='Save model after training')
    parser.add_argument('--load-model', action='store_true', help='Load existing model instead of training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num-episodes', type=int, default=2, help='Number of evaluation episodes')
    parser.add_argument('--episode-length', type=int, default=3000, help='Length of each episode')
    parser.add_argument('--x-vel', type=float, default=1.0, help='X velocity command')
    parser.add_argument('--y-vel', type=float, default=0.0, help='Y velocity command')
    parser.add_argument('--yaw-vel', type=float, default=0.0, help='Yaw velocity command')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    runner = ZBotRunner(args, logger)
    
    if args.load_model:
        runner.load_model()
    else:
        runner.train()
    
    runner.evaluate()


if __name__ == "__main__":
    main()
