---
sidebar_position: 3
id: timeline
title: Project Timeline
---

# ReinforceTactics RL Research - Task List

**Goal**: Submit paper to ICML by January 23, 2026

This document outlines the complete research and development timeline for the ReinforceTactics project, organized into weekly sprints with specific deliverables.

## Week 1 (Oct 22-28): Foundation - Headless Mode

### Day 1-2: Environment Setup & RL Gym Implementation
- [ ] Create `rl/gym_env.py` with complete Gymnasium implementation
- [ ] Test environment with random agent
- [ ] Verify no pygame dependencies in headless mode
- [ ] Benchmark performance (target: 1000+ steps/sec)

### Day 3-4: Action Space & Masking
- [ ] Complete `rl/action_space.py` with proper action masking
- [ ] Test action space coverage (ensure all game actions representable)
- [ ] Implement invalid action filtering
- [ ] Add action space statistics logging

### Day 5-7: Baseline Training
- [ ] Train flat PPO baseline (2-3M steps)
- [ ] Train flat DQN baseline (2-3M steps)
- [ ] Verify learning (>0% win rate vs SimpleBot)
- [ ] Set up TensorBoard logging
- [ ] Create `train.py` script
- [ ] Create `evaluate.py` script

---

## Week 2 (Oct 29 - Nov 4): Docker + GCP Setup

### Day 1-3: Docker Configuration
- [ ] Create Dockerfile with nvidia/cuda:11.8.0 base
- [ ] Install PyTorch, Stable-Baselines3, and game dependencies
- [ ] Create `docker-compose.yml` for multi-container experiments
- [ ] Test locally with CPU
- [ ] Push to Google Container Registry

### Day 4-5: GCP Setup
- [ ] Set up GCP project and billing
- [ ] Install and configure gcloud CLI
- [ ] Create VM instances (n1-standard-8 + NVIDIA T4/V100)
- [ ] Configure Cloud Storage for checkpoints/logs
- [ ] Set up instance templates for scaling
- [ ] Test single training run on GCP

### Day 6-7: Parallel Training Infrastructure
- [ ] Create training orchestration script for parallel seeds
- [ ] Set up experiment tracking (Weights & Biases or TensorBoard)
- [ ] Test parallel training (4-8 seeds simultaneously)
- [ ] Create cost monitoring dashboard
- [ ] Create `launch_gcp_training.sh` script

---

## Week 3 (Nov 5-11): Improved AI Opponents

### Day 1-3: Normal Difficulty Bot
- [ ] Implement `NormalBot` class with prioritized decision making
  - [ ] Capture nearly-complete structures first
  - [ ] Attack wounded enemies
  - [ ] Defend vulnerable structures
  - [ ] Expand economy
  - [ ] Train units based on composition
  - [ ] Move units toward strategic positions
- [ ] Test and tune to achieve ~60% win rate vs SimpleBot

### Day 4-7: Hard Difficulty Bot
- [ ] Choose approach (Heuristic-Based vs MCTS-lite)
- [ ] Implement `HardBot` class
  - [ ] Board state evaluation with scoring function
  - [ ] 2-3 turn lookahead planning
  - [ ] Unit positioning optimization
  - [ ] Economic optimization
  - [ ] Threat assessment
- [ ] Test and tune to achieve >80% win rate vs SimpleBot

---

## Week 4 (Nov 12-18): Evaluation & Statistics

### Day 1-3: Evaluation Framework
- [ ] Implement tournament system (round-robin between agents)
- [ ] Add statistical testing (bootstrapping, confidence intervals)
- [ ] Create visualization tools
  - [ ] Win rate curves
  - [ ] Game length distributions
- [ ] Create automated benchmarking script

### Day 4-7: Game Statistics & Analysis
- [ ] Implement detailed metrics logging
  - [ ] Turn-by-turn decisions
  - [ ] Unit composition over time
  - [ ] Territory control heatmaps
  - [ ] Economic efficiency
- [ ] Create replay analysis tools
- [ ] Generate baseline plots for paper

---

## Week 5 (Nov 19-25): Feudal RL - Architecture Design

### Day 1-2: Architecture Design
- [ ] Design Manager network architecture
  - [ ] Input: Full game state (grid, units, resources)
  - [ ] Output: Goal vector g_t
  - [ ] Update frequency: Every K steps (K=5-10)
- [ ] Design Worker network architecture
  - [ ] Input: Game state + current goal
  - [ ] Output: Primitive action
  - [ ] Update frequency: Every step
- [ ] Define goal space (start with spatial goals: x, y, goal_type)

### Day 3-7: Manager Implementation
- [ ] Implement `ManagerNetwork` class
  - [ ] CNN for spatial features
  - [ ] MLP for global features
  - [ ] Goal output head
- [ ] Design manager reward function
  - [ ] Win/loss (sparse)
  - [ ] Territory control improvement
  - [ ] Economic advantage
  - [ ] Structure capture
- [ ] Implement goal representation
  - [ ] Discrete spatial goals (grid cells)
  - [ ] One-hot goal type encoding
  - [ ] Goal embedding layer
- [ ] Test manager network in isolation

---

## Week 6 (Nov 26 - Dec 2): Feudal RL - Worker & Training

### Day 1-4: Worker Implementation
- [ ] Implement `WorkerNetwork` class
  - [ ] State encoder
  - [ ] Goal encoder
  - [ ] Combined processing
  - [ ] Policy head
- [ ] Design intrinsic reward for worker
  - [ ] Distance to goal decreased
  - [ ] Goal achievement bonus
  - [ ] Time penalty
- [ ] Implement goal conditioning (concatenation or FiLM layers)
- [ ] Test worker network in isolation

### Day 5-7: Joint Training
- [ ] Implement joint training algorithm
  - [ ] Manager goal selection
  - [ ] Worker action execution loop
  - [ ] Intrinsic reward computation
  - [ ] Worker updates
  - [ ] Manager updates
- [ ] Implement goal relabeling (hindsight for sample efficiency)
- [ ] Hyperparameter tuning
  - [ ] Manager update frequency (K)
  - [ ] Intrinsic/extrinsic reward balance
  - [ ] Learning rates
  - [ ] Network sizes
- [ ] Verify Feudal RL agent trains successfully

---

## Week 7 (Dec 3-9): Main Experiments

### Experiment 1: Sample Efficiency (MOST IMPORTANT)
- [ ] Set up parallel training on GCP (5-8 VMs)
- [ ] Train Feudal RL (5-10 seeds, 10M steps each)
- [ ] Train Flat PPO (5-10 seeds, 10M steps each)
- [ ] Train Flat DQN (5-10 seeds, 10M steps each)
- [ ] Use SimpleBot → NormalBot curriculum
- [ ] Log metrics: win rate, average return, game length
- [ ] Collect and aggregate results

### Experiment 2: Final Performance
- [ ] Train all methods to convergence (20-30M steps, 3-5 seeds)
- [ ] Test against SimpleBot
- [ ] Test against NormalBot
- [ ] Test against HardBot
- [ ] Collect final performance statistics

---

## Week 8 (Dec 10-16): Ablations & Generalization

### Ablation Studies
- [ ] Manager update frequency ablation (K=1, 5, 10, 20)
- [ ] Goal representation ablation (spatial vs abstract)
- [ ] Intrinsic reward design ablation
- [ ] Remove hierarchy ablation (compare to flat baseline)

### Generalization Tests
- [ ] Train on 20x20 maps, test on 30x30
- [ ] Train on 20x20 maps, test on 40x40
- [ ] Test transfer to different starting resources
- [ ] Zero-shot testing vs different opponent types

### Analysis
- [ ] Statistical significance testing on all results
- [ ] Generate all experimental plots
- [ ] Create results tables

---

## Week 9 (Dec 17-23): Paper Writing - Core Sections

### Introduction (2 pages)
- [ ] Write hook about RL in strategy games
- [ ] Define problem statement
- [ ] Describe solution approach
- [ ] List contributions (3-4 main points)

### Related Work (1-1.5 pages)
- [ ] RL in strategy games section
- [ ] Hierarchical RL section
- [ ] Action abstraction section
- [ ] Position your work clearly

### Method (2-2.5 pages)
- [ ] Game formalization (MDP definition)
- [ ] Observation space description
- [ ] Action space description
- [ ] Reward function description
- [ ] Feudal RL architecture (Manager + Worker)
- [ ] Training algorithm with pseudocode
- [ ] Network architectures
- [ ] Hyperparameters table

---

## Week 10 (Dec 24-30): Paper Writing - Results & Discussion

### Experiments (2 pages)
- [ ] Write experimental setup section
- [ ] Describe baselines
- [ ] Create sample efficiency plots
- [ ] Create final performance table
- [ ] Add statistical significance tests
- [ ] Write results narrative

### Analysis (1 page)
- [ ] Present ablation study results
- [ ] Present generalization test results
- [ ] Qualitative analysis (learned goals)
- [ ] Discuss failure cases

### Discussion & Conclusion (1 page)
- [ ] Summarize contributions
- [ ] Discuss limitations
- [ ] Propose future work
- [ ] Add broader impact statement

---

## Week 11 (Dec 31 - Jan 6): Refinement

### Day 1-2: Publication-Quality Figures
- [ ] Style all plots with matplotlib/seaborn
- [ ] Add error bars to all plots
- [ ] Add clear legends and axis labels
- [ ] Export at high resolution (300 DPI)
- [ ] Ensure consistent styling across all figures

### Day 3-4: Additional Experiments (if needed)
- [ ] Identify gaps in results
- [ ] Run additional experiments to strengthen weak sections
- [ ] Add any requested ablations from self-review

### Day 5-7: Get Feedback
- [ ] Self-review draft with fresh eyes
- [ ] Ask colleagues/advisors for feedback (if available)
- [ ] Check against ICML review criteria
- [ ] Incorporate feedback into draft v2

---

## Week 12 (Jan 7-13): Polish

### Day 1-2: Writing Quality
- [ ] Remove redundancy throughout paper
- [ ] Improve clarity in all sections
- [ ] Check flow between sections
- [ ] Verify all claims are supported by results
- [ ] Fix grammar and typos

### Day 3-4: Appendix
- [ ] Add detailed hyperparameters table
- [ ] Include additional experimental results
- [ ] Add pseudocode for all algorithms
- [ ] Create network architecture diagrams
- [ ] Add any supplementary material

### Day 5-7: Final Checks
- [ ] Apply ICML formatting (use official LaTeX template)
- [ ] Format all references correctly
- [ ] Verify page limit compliance (8 pages + unlimited appendix)
- [ ] Proofread entire paper 3+ times
- [ ] Check all figures/tables render correctly
- [ ] Verify all citations are correct

---

## Week 13 (Jan 14-23): Submission

### Day 1-3: Final Polish
- [ ] Last round of edits
- [ ] Fix any formatting issues
- [ ] Verify all ICML submission requirements
- [ ] Generate final PDF

### Day 4: Buffer Day
- [ ] Handle any unexpected issues
- [ ] Final sanity checks

### Day 5: SUBMISSION 🎉
- [ ] Submit to ICML by January 23, 2026 deadline!

---

## Immediate Action Items (This Week - Oct 22-28)

- [ ] **Today**: Create `rl/gym_env.py`
- [ ] **Tomorrow**: Set up GCP project and gcloud CLI
- [ ] **Day 3**: Test baseline training with `python train.py --algorithm ppo --timesteps 100000 --opponent bot`
- [ ] **Day 4-5**: Create Dockerfile
- [ ] **Day 6**: Set up experiment tracking (Weights & Biases)

---

## Resources & Scripts Needed

- [ ] Complete `rl/gym_env.py` implementation
- [ ] `Dockerfile` for GCP training
- [ ] `docker-compose.yml` for local testing
- [ ] `launch_gcp_training.sh` script
- [ ] `requirements.txt` with all dependencies
- [ ] Training scripts (`train.py`, `evaluate.py`)
- [ ] Analysis scripts for generating plots

---

## Budget & Cost Tracking

- **GCP Free Tier**: $300 credit
- **Expected Total Cost**: $200-500 for all experiments
- **Cost per GPU hour**: ~$0.50-1.50 (T4/V100)
- [ ] Monitor GCP spending weekly
- [ ] Use preemptible instances for cost savings
- [ ] Stop instances when not actively training

---

## Progress Tracking

Use this section to track overall progress:

- **Current Week**: Week 1 - Foundation
- **Completion Status**: 0%
- **Days Until Deadline**: ~90 days
- **Critical Path**: Weeks 5-8 (Feudal RL implementation and experiments)
