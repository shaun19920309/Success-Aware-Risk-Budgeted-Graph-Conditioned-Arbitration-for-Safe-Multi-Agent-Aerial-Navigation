import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.use_lagrangian = bool(getattr(self.all_args, "use_lagrangian", False))
        self.lagrangian_cost_type = str(getattr(self.all_args, "lagrangian_cost_type", "hybrid"))
        self.lagrangian_cost_limit = float(getattr(self.all_args, "lagrangian_cost_limit", 0.02))
        self.lagrangian_lr = float(getattr(self.all_args, "lagrangian_lr", 0.05))
        self.lagrangian_multiplier = float(getattr(self.all_args, "lagrangian_init", 0.0))
        self.lagrangian_max = float(getattr(self.all_args, "lagrangian_max", 20.0))
        self.resume_step = int(os.environ.get("ONPOLICY_RESUME_STEP", "0") or 0)
        step_stride = self.episode_length * self.n_rollout_threads
        if self.resume_step < 0 or self.resume_step >= int(self.num_env_steps):
            raise ValueError(
                f"ONPOLICY_RESUME_STEP must be in [0, {self.num_env_steps}), got {self.resume_step}."
            )
        if self.resume_step % step_stride != 0:
            raise ValueError(
                f"ONPOLICY_RESUME_STEP={self.resume_step} is not aligned to rollout stride {step_stride}."
            )
        self.milestone_steps = tuple(
            sorted(
                {
                    int(step)
                    for step in getattr(self.all_args, "milestone_steps", [])
                    if int(step) > 0
                }
            )
        )
        run_dir = Path(self.save_dir).parent
        self.saved_milestone_steps = {
            step
            for step in self.milestone_steps
            if (run_dir / "milestones" / f"step{step}" / "milestone_manifest.json").is_file()
        }
        self._restore_training_state_if_available()

    def _training_state(self, completed_steps):
        optimizers = {}
        for name in ("actor_optimizer", "critic_optimizer", "optimizer"):
            optimizer = getattr(self.trainer.policy, name, None)
            if optimizer is not None:
                optimizers[name] = optimizer.state_dict()
        value_normalizer = getattr(self.trainer, "value_normalizer", None)
        state = {
            "format_version": 1,
            "algorithm": self.algorithm_name,
            "training_seed": int(self.all_args.seed),
            "completed_steps": int(completed_steps),
            "optimizers": optimizers,
            "value_normalizer": value_normalizer.state_dict()
            if value_normalizer is not None
            else None,
            "lagrangian_multiplier": self.lagrangian_multiplier,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _save_training_state(self, model_dir, completed_steps):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._training_state(completed_steps),
            model_dir / "training_state.pt",
        )

    def _restore_training_state_if_available(self):
        if self.resume_step == 0:
            return
        state_path_raw = os.environ.get("ONPOLICY_RESUME_TRAINING_STATE", "").strip()
        if state_path_raw:
            state_path = Path(state_path_raw).expanduser().resolve()
            if not state_path.is_file():
                raise FileNotFoundError(f"Resume training state not found: {state_path}")
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            if int(state.get("completed_steps", -1)) != self.resume_step:
                raise ValueError(
                    f"Training-state step {state.get('completed_steps')} does not match "
                    f"ONPOLICY_RESUME_STEP={self.resume_step}."
                )
            for name, optimizer_state in state.get("optimizers", {}).items():
                optimizer = getattr(self.trainer.policy, name, None)
                if optimizer is None:
                    raise ValueError(f"Checkpoint contains unavailable optimizer {name}.")
                optimizer.load_state_dict(optimizer_state)
            value_normalizer = getattr(self.trainer, "value_normalizer", None)
            if state.get("value_normalizer") is not None:
                if value_normalizer is None:
                    raise ValueError("Checkpoint contains a value normalizer but trainer does not.")
                value_normalizer.load_state_dict(state["value_normalizer"])
            self.lagrangian_multiplier = float(
                state.get("lagrangian_multiplier", self.lagrangian_multiplier)
            )
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            torch.set_rng_state(state["torch_random_state"].cpu())
            if torch.cuda.is_available() and "torch_cuda_random_state_all" in state:
                torch.cuda.set_rng_state_all([rng.cpu() for rng in state["torch_cuda_random_state_all"]])
            print(f"Restored full on-policy training state: {state_path}", flush=True)
            return

        if self.use_lagrangian:
            multiplier_raw = os.environ.get(
                "ONPOLICY_RESUME_LAGRANGIAN_MULTIPLIER", ""
            ).strip()
            if not multiplier_raw:
                raise RuntimeError(
                    "A Lagrangian resume requires ONPOLICY_RESUME_LAGRANGIAN_MULTIPLIER "
                    "when no full training state is available."
                )
            self.lagrangian_multiplier = float(multiplier_raw)
            if not 0.0 <= self.lagrangian_multiplier <= self.lagrangian_max:
                raise ValueError(
                    f"Recovered Lagrangian multiplier {self.lagrangian_multiplier} is out of range."
                )
        print(
            "Resuming from legacy weight-only checkpoint; optimizer and value-normalizer "
            "states were not available.",
            flush=True,
        )

    def _save_milestone(self, requested_steps, actual_steps):
        run_dir = Path(self.save_dir).parent
        milestone_dir = run_dir / "milestones" / f"step{requested_steps}"
        model_dir = milestone_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.algorithm_name in {"mat", "mat_dec"}:
            self.policy.save(model_dir, 0)
        else:
            torch.save(
                self.trainer.policy.actor.state_dict(),
                model_dir / "actor.pt",
            )
            torch.save(
                self.trainer.policy.critic.state_dict(),
                model_dir / "critic.pt",
            )
        self._save_training_state(model_dir, actual_steps)
        config_path = run_dir / "config.json"
        if config_path.is_file():
            shutil.copy2(config_path, milestone_dir / "config.json")
        manifest = {
            "requested_steps": int(requested_steps),
            "actual_steps": int(actual_steps),
            "algorithm": self.algorithm_name,
            "training_seed": int(self.all_args.seed),
        }
        (milestone_dir / "milestone_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        self.saved_milestone_steps.add(int(requested_steps))
        print(
            f"Saved milestone step{requested_steps} at actual timestep {actual_steps}: "
            f"{milestone_dir}",
            flush=True,
        )

    def _save_due_milestones(self, total_num_steps, final_update):
        for requested_steps in self.milestone_steps:
            if requested_steps in self.saved_milestone_steps:
                continue
            due = total_num_steps >= requested_steps
            final_due = final_update and requested_steps == int(self.num_env_steps)
            if due or final_due:
                self._save_milestone(requested_steps, total_num_steps)

    def _extract_safety_costs(self, infos, rewards):
        costs = np.zeros_like(rewards, dtype=np.float32)
        for env_i, env_info in enumerate(infos):
            if isinstance(env_info, np.ndarray):
                if env_info.ndim == 0:
                    env_info = env_info.item()
                    agent_infos = [env_info]
                else:
                    agent_infos = env_info.reshape(-1).tolist()
            elif isinstance(env_info, (list, tuple)):
                agent_infos = env_info
            elif isinstance(env_info, dict):
                agent_infos = [env_info]
            else:
                continue

            for agent_i, agent_info in enumerate(agent_infos):
                if env_i >= costs.shape[0] or agent_i >= costs.shape[1] or not isinstance(agent_info, dict):
                    continue
                reward_parts = agent_info.get("rewards", {})
                if not isinstance(reward_parts, dict):
                    continue

                raw_quad = float(reward_parts.get("rewraw_quadcol", 0.0))
                raw_obst = float(reward_parts.get("rewraw_quadcol_obstacle", 0.0))
                proximity = float(reward_parts.get("rew_proximity", 0.0))

                cost = 0.0
                if self.lagrangian_cost_type in {"collision", "hybrid"}:
                    cost += float(raw_quad < 0.0 or raw_obst < 0.0)
                if self.lagrangian_cost_type in {"proximity", "hybrid"}:
                    cost += max(0.0, -proximity)
                costs[env_i, agent_i, 0] = cost
        return costs

    def _apply_lagrangian_penalty(self, rewards, infos):
        if not self.use_lagrangian:
            return rewards, np.zeros_like(rewards, dtype=np.float32)
        costs = self._extract_safety_costs(infos, rewards)
        penalized_rewards = rewards - self.lagrangian_multiplier * costs
        return penalized_rewards.astype(np.float32), costs

    def _update_lagrangian_multiplier(self, cost_mean):
        if not self.use_lagrangian:
            return
        self.lagrangian_multiplier += self.lagrangian_lr * (float(cost_mean) - self.lagrangian_cost_limit)
        self.lagrangian_multiplier = float(np.clip(self.lagrangian_multiplier, 0.0, self.lagrangian_max))

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        start_episode = self.resume_step // (
            self.episode_length * self.n_rollout_threads
        )
        if self.resume_step:
            print(
                f"Continuing at update {start_episode}/{episodes}, cumulative steps "
                f"{self.resume_step}/{self.num_env_steps}.",
                flush=True,
            )

        for episode in range(start_episode, episodes):
            rollout_costs = []
            rollout_raw_rewards = []
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                rollout_raw_rewards.append(rewards.copy())
                rewards, safety_costs = self._apply_lagrangian_penalty(rewards, infos)
                rollout_costs.append(safety_costs.copy())

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            if self.use_lagrangian:
                cost_mean = float(np.mean(rollout_costs)) if rollout_costs else 0.0
                raw_reward_mean = float(np.mean(rollout_raw_rewards) * self.episode_length) if rollout_raw_rewards else 0.0
                self._update_lagrangian_multiplier(cost_mean)
                train_infos["lagrangian/cost_mean"] = cost_mean
                train_infos["lagrangian/cost_limit"] = self.lagrangian_cost_limit
                train_infos["lagrangian/multiplier"] = self.lagrangian_multiplier
                train_infos["lagrangian/raw_average_episode_rewards"] = raw_reward_mean
                train_infos["lagrangian/penalized_average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                self._save_training_state(self.save_dir, total_num_steps)
            self._save_due_milestones(
                total_num_steps,
                final_update=episode == episodes - 1,
            )

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int((total_num_steps - self.resume_step) / (end - start))))

                env_infos = {}
                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions
            low = self.envs.action_space[0].low
            high = self.envs.action_space[0].high
            actions_env = np.clip(actions_env, low, high).astype(np.float32)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Box':
                eval_actions_env = eval_actions
                low = self.eval_envs.action_space[0].low
                high = self.eval_envs.action_space[0].high
                eval_actions_env = np.clip(eval_actions_env, low, high).astype(np.float32)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                    low = envs.action_space[0].low
                    high = envs.action_space[0].high
                    actions_env = np.clip(actions_env, low, high).astype(np.float32)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
