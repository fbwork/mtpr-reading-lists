"""
Main script to run TCH-PR experiments.

Usage:
    python run_tchpr.py --setup          # Setup caches (run once)
    python run_tchpr.py --query "statistical parsing" --top-k 10
    python run_tchpr.py --ablation       # Run ablation study
"""

import argparse
import time
from components.tchpr import TCHPR, setup_tchpr


def main():
    parser = argparse.ArgumentParser(description="TCH-PR Experiments")
    parser.add_argument('--setup', action='store_true', help='Setup caches')
    parser.add_argument('--query', type=str, help='Query text')
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--lda-topics', type=int, default=50, help='Number of LDA topics')
    
    args = parser.parse_args()
    
    if args.setup:
        print("Setting up TCH-PR (this may take 10-30 minutes)...")
        tchpr = setup_tchpr(lda_no_topics=args.lda_topics, fresh=True)
        tchpr.save_caches()
        print("Setup complete!")
    
    elif args.query:
        print(f"Loading TCH-PR...")
        tchpr = TCHPR(lda_no_topics=args.lda_topics)
        tchpr.setup(fresh=False)
        
        print(f"\nQuery: {args.query}")
        print("-" * 60)
        
        start = time.time()
        results = tchpr.recommend(args.query, top_k=args.top_k)
        elapsed = time.time() - start
        
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"{rank:2d}. Paper {doc_id:5d}  Score: {score:.4f}")
        
        print(f"\nQuery time: {elapsed:.3f}s")
    
    elif args.ablation:
        print("Running ablation study...")
        # TODO: Implement ablation
        print("Ablation study not yet implemented.")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()