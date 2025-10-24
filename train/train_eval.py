"""Training and evaluation functions for tactical combat RL agents"""

from datetime import datetime

# Try to import numpy with guards
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import SB3 with guards
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# Try to import RLlib with guards
try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.rllib.algorithms.a3c import A3CConfig
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

from reinforcetactics.entities import Team, CurriculumStage
from reinforcetactics.leaderboard import Leaderboard
from reinforcetactics.model_manager import ModelManager

# Import environment classes conditionally
try:
    from reinforcetactics.envs import TacticalCombatGymEnv, TacticalCombatRLlibEnv
except ImportError:
    TacticalCombatGymEnv = None
    TacticalCombatRLlibEnv = None

# Import curriculum conditionally
try:
    from reinforcetactics.curriculum import CurriculumManager
except ImportError:
    CurriculumManager = None


# Training Functions
def train_sb3_agent(algorithm: str = "ppo", total_timesteps: int = 100000, 
                    use_curriculum: bool = True, model_name: str = None):
    """Train with Stable-Baselines3"""
    if not SB3_AVAILABLE:
        print("❌ Stable-Baselines3 not installed.")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 Training {algorithm.upper()} Agent with Stable-Baselines3")
    print(f"{'='*60}\n")
    
    curriculum_manager = CurriculumManager() if use_curriculum else None
    initial_stage = curriculum_manager.get_current_stage() if use_curriculum else CurriculumStage.MEDIUM
    
    def make_env():
        env = TacticalCombatGymEnv(curriculum_stage=initial_stage)
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    if algorithm.lower() == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    elif algorithm.lower() == "dqn":
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    else:
        print(f"❌ Unknown algorithm: {algorithm}")
        return
    
    print("🎮 Training started...")
    model.learn(total_timesteps=total_timesteps)
    print("✅ Training completed!")
    
    model_manager = ModelManager()
    if model_name is None:
        model_name = f"sb3_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metadata = {
        "algorithm": algorithm,
        "framework": "stable-baselines3",
        "total_timesteps": total_timesteps,
        "curriculum_learning": use_curriculum,
        "timestamp": datetime.now().isoformat()
    }
    
    model_manager.save_model(model, model_name, metadata)
    
    print("\n📊 Evaluating model...")
    eval_results = evaluate_sb3_agent(model)
    
    leaderboard = Leaderboard()
    leaderboard.add_entry(
        model_name=model_name,
        algorithm=algorithm,
        framework="SB3",
        win_rate=eval_results["win_rate"],
        avg_reward=eval_results["avg_reward"],
        avg_damage=eval_results["avg_damage"],
        avg_kills=eval_results["avg_kills"],
        curriculum_stage="MEDIUM",
        total_episodes=total_timesteps
    )
    
    return model


def train_rllib_agent(algorithm: str = "PPO", total_timesteps: int = 100000, 
                      multiagent: bool = False, model_name: str = None):
    """Train with Ray RLlib"""
    if not RLLIB_AVAILABLE:
        print("❌ Ray RLlib not installed. Install with: pip install 'ray[rllib]' torch")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 Training {algorithm} Agent with Ray RLlib")
    print(f"Multi-Agent Mode: {'✅ Enabled' if multiagent else '❌ Disabled'}")
    print(f"{'='*60}\n")
    
    ray.init(ignore_reinit_error=True)
    
    # Configure algorithm
    if algorithm.upper() == "PPO":
        config = PPOConfig()
    elif algorithm.upper() == "DQN":
        config = DQNConfig()
    elif algorithm.upper() == "A3C":
        config = A3CConfig()
    else:
        print(f"❌ Unknown algorithm: {algorithm}")
        return
    
    # Setup environment
    config = config.environment(
        TacticalCombatRLlibEnv,
        env_config={
            "curriculum_stage": CurriculumStage.MEDIUM,
            "multiagent": multiagent
        }
    )
    
    # Training config
    config = config.training(
        lr=0.0003,
        train_batch_size=4000,
    )
    
    # Resources config
    config = config.resources(
        num_gpus=0,  # Set to 1 if GPU available
    )
    
    # Rollout config
    config = config.rollouts(
        num_rollout_workers=2,
    )
    
    # Build algorithm
    algo = config.build()
    
    print("🎮 Training started...")
    iterations = total_timesteps // 4000  # Approximate
    
    for i in range(iterations):
        result = algo.train()
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}: "
                  f"Reward={result['episode_reward_mean']:.2f}, "
                  f"Episode Length={result['episode_len_mean']:.1f}")
        
        # Save checkpoint periodically
        if (i + 1) % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    print("✅ Training completed!")
    
    # Save final model
    if model_name is None:
        model_name = f"rllib_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    checkpoint_dir = algo.save()
    
    # Save metadata
    model_manager = ModelManager()
    metadata = {
        "algorithm": algorithm,
        "framework": "rllib",
        "total_timesteps": total_timesteps,
        "multiagent": multiagent,
        "checkpoint_dir": checkpoint_dir,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = model_manager.models_dir + f"/{model_name}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model and metadata saved: {checkpoint_dir}")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = evaluate_rllib_agent(algo)
    
    leaderboard = Leaderboard()
    leaderboard.add_entry(
        model_name=model_name,
        algorithm=algorithm,
        framework="RLlib",
        win_rate=eval_results["win_rate"],
        avg_reward=eval_results["avg_reward"],
        avg_damage=eval_results["avg_damage"],
        avg_kills=eval_results["avg_kills"],
        curriculum_stage="MEDIUM",
        total_episodes=total_timesteps
    )
    
    ray.shutdown()
    return algo


def evaluate_sb3_agent(model, episodes: int = 100):
    """Evaluate SB3 agent"""
    print(f"\n🧪 Evaluating SB3 agent over {episodes} episodes...")
    
    env = TacticalCombatGymEnv(curriculum_stage=CurriculumStage.MEDIUM)
    
    wins = 0
    total_rewards = []
    total_damage = []
    total_kills = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        winner = env.game_state.check_victory()
        if winner == Team.PLAYER:
            wins += 1
        
        metrics = env.game_state.get_metrics(Team.PLAYER)
        total_rewards.append(episode_reward)
        total_damage.append(metrics["damage_dealt"])
        total_kills.append(metrics["units_killed"])
    
    results = {
        "win_rate": wins / episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_damage": np.mean(total_damage),
        "avg_kills": np.mean(total_kills)
    }
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results:")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Avg Reward:      {results['avg_reward']:.2f}")
    print(f"{'='*60}\n")
    
    return results


def evaluate_rllib_agent(algo, episodes: int = 100):
    """Evaluate RLlib agent"""
    print(f"\n🧪 Evaluating RLlib agent over {episodes} episodes...")
    
    env = TacticalCombatRLlibEnv(config={"curriculum_stage": CurriculumStage.MEDIUM})
    
    wins = 0
    total_rewards = []
    total_damage = []
    total_kills = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        winner = env.game_state.check_victory()
        if winner == Team.PLAYER:
            wins += 1
        
        metrics = env.game_state.get_metrics(Team.PLAYER)
        total_rewards.append(episode_reward)
        total_damage.append(metrics["damage_dealt"])
        total_kills.append(metrics["units_killed"])
    
    results = {
        "win_rate": wins / episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_damage": np.mean(total_damage),
        "avg_kills": np.mean(total_kills)
    }
    
    print(f"\n{'='*60}")
    print(f"📊 RLlib Evaluation Results:")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Avg Reward:      {results['avg_reward']:.2f}")
    print(f"{'='*60}\n")
    
    return results
