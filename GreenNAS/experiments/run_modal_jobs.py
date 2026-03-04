"""Fire-and-forget launcher for both Modal training jobs."""
import modal

app = modal.App("greennas-launcher")

@app.local_entrypoint()
def main():
    # Spawn both jobs as fire-and-forget
    train_fn = modal.Function.from_name('greennas-real-training', 'train_topk')
    val_fn = modal.Function.from_name('greennas-real-training', 'validate_l4')
    
    print("Spawning T4 top-K training...")
    h1 = train_fn.spawn()
    print(f"  -> {h1.object_id}")
    
    print("Spawning L4 validation...")
    h2 = val_fn.spawn()
    print(f"  -> {h2.object_id}")
    
    print("\nBoth jobs spawned. Waiting for L4 (shorter job) to complete...")
    # Wait for L4 result (should be ~30 min)
    try:
        result = h2.get(timeout=3600)  # 1 hour timeout
        print(f"\nL4 validation complete!")
        print(f"  Correlations: {result.get('correlations', 'N/A')}")
    except Exception as e:
        print(f"L4 job error: {e}")
    
    print("\nWaiting for T4 training to complete...")
    try:
        result = h1.get(timeout=14400)  # 4 hour timeout
        print(f"\nT4 training complete! {len(result)} architectures trained.")
    except Exception as e:
        print(f"T4 job error: {e}")
    
    print("\nDone! Check: modal volume ls greennas-results /")
