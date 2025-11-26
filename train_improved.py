#!/usr/bin/env python3
"""
Training script for improved PCL agent.

Usage:
    # Run improved PCL training
    python train_improved.py --method improved-pcl --iterations 500
    
    # Run imitation learning (behavioral cloning)
    python train_improved.py --method bc --n-expert 20
    
    # Run imitation learning with PCL fine-tuning
    python train_improved.py --method bc+pcl --n-expert 20 --finetune-iterations 100
    
    # Run DAgger
    python train_improved.py --method dagger --dagger-rounds 10
    
    # Run diagnostics on existing model
    python train_improved.py --diagnose --model best_pcl_agent.pt
"""

import torch
import numpy as np
import argparse
import time

from data_loader import load_env, create_synthetic_env


def main():
    parser = argparse.ArgumentParser(description="Train improved PCL for aircraft scheduling")
    
    # Mode selection
    parser.add_argument('--method', type=str, default='improved-pcl',
                       choices=['improved-pcl', 'bc', 'bc+pcl', 'dagger'],
                       help='Training method to use')
    parser.add_argument('--diagnose', action='store_true',
                       help='Run diagnostics on existing model instead of training')
    parser.add_argument('--model', type=str, default='best_pcl_agent.pt',
                       help='Model path for diagnostics')
    
    # Common training parameters
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=15.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # PCL-specific parameters
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--entropy', type=float, default=0.1)
    parser.add_argument('--shaping-coeff', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=50)
    
    # Replay buffer parameters
    parser.add_argument('--buffer-capacity', type=int, default=50000)
    parser.add_argument('--segment-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gradient-steps', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    
    # Imitation learning parameters
    parser.add_argument('--n-expert', type=int, default=20,
                       help='Number of expert trajectories for BC')
    parser.add_argument('--bc-epochs', type=int, default=100)
    parser.add_argument('--finetune-iterations', type=int, default=100)
    parser.add_argument('--dagger-rounds', type=int, default=10)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load environment
    print("="*80)
    print("AIRCRAFT MAINTENANCE SCHEDULING")
    print("="*80)
    
    try:
        print("\nLoading data from Excel...")
        env = load_env("Sep 25 2017 (Case Study 2018-2021).xlsx")
    except Exception as e:
        print(f"Could not load Excel: {e}")
        print("\nUsing synthetic environment...")
        env = create_synthetic_env(n_aircraft=45, horizon_days=1461)
    
    config = env.config
    
    print(f"\nEnvironment:")
    print(f"  - Aircraft: {env.n_aircraft}")
    print(f"  - Horizon: {env.horizon} days (~{env.horizon/365:.1f} years)")
    print(f"  - Action space: {env.n_actions}")
    
    # Diagnostics mode
    if args.diagnose:
        print(f"\n*** Running diagnostics on {args.model} ***")
        from diagnose_pcl import (
            load_pcl_agent,
            diagnose_trajectory,
            plot_diagnostics,
            analyze_scheduling_timing,
            compare_pcl_vs_greedy_detailed
        )
        
        # Try enhanced first, fall back to basic
        try:
            agent = load_pcl_agent(args.model, env, enhanced=True)
            enhanced = True
        except:
            agent = load_pcl_agent(args.model, env, enhanced=False)
            enhanced = False
        
        print(f"  Loaded model (enhanced={enhanced})")
        
        diagnostics = diagnose_trajectory(agent, env, config, enhanced=enhanced)
        plot_diagnostics(diagnostics, 'pcl_diagnostics.png')
        analyze_scheduling_timing(diagnostics, config)
        compare_pcl_vs_greedy_detailed(agent, env, config, enhanced=enhanced)
        return
    
    # Training mode
    if args.method == 'improved-pcl':
        print(f"\n*** Improved PCL Training ***")
        from pcl_improved import (
            ImprovedPCLAgent,
            get_enhanced_state_dim,
            train_pcl_improved,
            evaluate_against_baselines
        )
        
        state_dim = get_enhanced_state_dim(env.n_aircraft)
        agent = ImprovedPCLAgent(
            state_dim=state_dim,
            n_actions=env.n_actions,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            num_layers=3
        )
        
        print(f"\nAgent: {sum(p.numel() for p in agent.parameters()):,} parameters")
        print(f"State dim: {state_dim}")
        
        train_pcl_improved(
            agent=agent,
            env=env,
            config=config,
            num_iterations=args.iterations,
            lr=args.lr,
            n_steps=args.n_steps,
            entropy_coeff=args.entropy,
            eval_every=args.eval_every,
            use_shaping=True,
            shaping_coeff=args.shaping_coeff,
            save_path='best_pcl_improved.pt',
            verbose=True,
            # Replay buffer settings
            buffer_capacity=args.buffer_capacity,
            segment_length=args.segment_length,
            batch_size=args.batch_size,
            gradient_steps_per_iter=args.gradient_steps,
            warmup_trajectories=args.warmup,
        )
        
        # Load best and evaluate
        checkpoint = torch.load('best_pcl_improved.pt', weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        evaluate_against_baselines(agent, env, config, n_episodes=5)
        
    elif args.method == 'bc':
        print(f"\n*** Behavioral Cloning ***")
        from pcl_improved import ImprovedPCLAgent, get_enhanced_state_dim
        from imitation_learning import (
            collect_expert_trajectories,
            behavioral_cloning,
            evaluate_imitation_agent
        )
        
        state_dim = get_enhanced_state_dim(env.n_aircraft)
        agent = ImprovedPCLAgent(
            state_dim=state_dim,
            n_actions=env.n_actions,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            num_layers=3
        )
        
        print(f"\nAgent: {sum(p.numel() for p in agent.parameters()):,} parameters")
        
        expert_data = collect_expert_trajectories(env, config, args.n_expert)
        behavioral_cloning(agent, expert_data, args.bc_epochs)
        
        torch.save({
            'model_state_dict': agent.state_dict(),
            'method': 'bc',
        }, 'imitation_bc.pt')
        
        evaluate_imitation_agent(agent, env, config, n_episodes=5)
        
    elif args.method == 'bc+pcl':
        print(f"\n*** BC + PCL Fine-tuning ***")
        from pcl_improved import ImprovedPCLAgent, get_enhanced_state_dim
        from imitation_learning import (
            collect_expert_trajectories,
            behavioral_cloning,
            finetune_with_pcl,
            evaluate_imitation_agent
        )
        
        state_dim = get_enhanced_state_dim(env.n_aircraft)
        agent = ImprovedPCLAgent(
            state_dim=state_dim,
            n_actions=env.n_actions,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            num_layers=3
        )
        
        print(f"\nAgent: {sum(p.numel() for p in agent.parameters()):,} parameters")
        
        # Phase 1: BC
        expert_data = collect_expert_trajectories(env, config, args.n_expert)
        behavioral_cloning(agent, expert_data, args.bc_epochs)
        
        # Phase 2: PCL fine-tuning
        finetune_with_pcl(
            agent, env, config,
            num_iterations=args.finetune_iterations,
            save_path='best_bc_pcl.pt'
        )
        
        # Load best and evaluate
        checkpoint = torch.load('best_bc_pcl.pt', weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        evaluate_imitation_agent(agent, env, config, n_episodes=5)
        
    elif args.method == 'dagger':
        print(f"\n*** DAgger Training ***")
        from pcl_improved import ImprovedPCLAgent, get_enhanced_state_dim
        from imitation_learning import dagger_training, evaluate_imitation_agent
        
        state_dim = get_enhanced_state_dim(env.n_aircraft)
        agent = ImprovedPCLAgent(
            state_dim=state_dim,
            n_actions=env.n_actions,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
            num_layers=3
        )
        
        print(f"\nAgent: {sum(p.numel() for p in agent.parameters()):,} parameters")
        
        dagger_training(
            agent, env, config,
            num_rounds=args.dagger_rounds,
            trajectories_per_round=5
        )
        
        torch.save({
            'model_state_dict': agent.state_dict(),
            'method': 'dagger',
        }, 'imitation_dagger.pt')
        
        evaluate_imitation_agent(agent, env, config, n_episodes=5)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

