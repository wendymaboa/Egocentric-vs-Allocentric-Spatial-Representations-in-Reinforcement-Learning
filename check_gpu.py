"""
Check GPU capacity for parallel training
"""
import torch
import subprocess

print("="*60)
print("GPU CAPACITY CHECK")
print("="*60)

if torch.cuda.is_available():
    print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total VRAM: {total_memory:.2f} GB")
    
    # Current usage
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"  Currently allocated: {allocated:.2f} GB")
    print(f"  Currently reserved: {reserved:.2f} GB")
    print(f"  Available: {total_memory - reserved:.2f} GB")
    
    # Recommendations
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATIONS")
    print("="*60)
    
    if total_memory >= 12:
        print("✓ You can train 2-3 agents in parallel")
        print("  Recommendation: Use separate terminals with CUDA_VISIBLE_DEVICES")
    elif total_memory >= 8:
        print("✓ You can train 2 agents in parallel (tight)")
        print("  Recommendation: Train sequentially or use 2 parallel with monitoring")
    elif total_memory >= 6:
        print("⚠ Train 1 agent at a time recommended")
        print("  You might fit 2 if using small environments (Empty-5x5)")
    else:
        print("⚠ Train 1 agent at a time (limited VRAM)")
    
    print("\nEstimated VRAM per agent:")
    print("  - Empty-5x5:    ~1-2 GB")
    print("  - Empty-8x8:    ~1.5-2.5 GB")
    print("  - DoorKey-5x5:  ~2-3 GB")
    print("  - GoToObject:   ~2-3 GB")
    
    print("\n" + "="*60)
    print("PARALLEL TRAINING METHODS")
    print("="*60)
    print("\n1. Different GPUs (if you have multiple):")
    print("   Terminal 1: CUDA_VISIBLE_DEVICES=0 python train_sb3.py ...")
    print("   Terminal 2: CUDA_VISIBLE_DEVICES=1 python train_sb3.py ...")
    
    print("\n2. Same GPU (if enough VRAM):")
    print("   Terminal 1: python train_sb3.py --env_key Empty-5x5 --agent_type ppo --egocentric")
    print("   Terminal 2: python train_sb3.py --env_key Empty-5x5 --agent_type ppo")
    
    print("\n3. Sequential with different seeds (safest):")
    print("   python train_sb3.py --seeds 0 1 2  # Trains all seeds sequentially")
    
else:
    print("\n✗ No GPU detected!")
    print("  Training will be VERY slow on CPU")
    print("  Recommendation: Use a GPU or cloud service (Colab, AWS, etc.)")

print("\n" + "="*60)