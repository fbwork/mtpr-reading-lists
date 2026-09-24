# --- File: main.py ---
import argparse
import sys
from pathlib import Path

# Add components to path
sys.path.insert(0, str(Path(__file__).parent))

from components import database, atr, lda, pagerank, evaluation, query, baselines
from components.topic_model_modes import TopicModelModes
from components.prmodes import PRModes

DESCRIPTION = 'Recommends list of papers for researchers looking for somewhere to start.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparser = parser.add_subparsers(dest='module', required=True)

    # Define all subparsers
    database_module = database.define_subparser(subparser)
    atr_module = atr.define_subparser(subparser)
    lda_module = lda.define_subparser(subparser)
    pagerank_module = pagerank.define_subparser(subparser)
    evaluation_module = evaluation.define_subparser(subparser)
    query_module = query.define_subparser(subparser)
    baselines_module = baselines.define_subparser(subparser)

    args = parser.parse_args()

    # Route to appropriate module
    if args.module == database_module:
        database.setup(fresh=args.fresh, force=getattr(args, 'force', False))

    elif args.module == atr_module:
        atr.setup(
            fresh=args.fresh,
            force=getattr(args, 'force', False),
            chunk_size=getattr(args, 'chunk_size', 100)
        )

    elif args.module == lda_module:
        lda.setup(
            lda_no_topics=args.lda_no_topics,
            bertopic_no_topics=0,   # Force flat LDA only
            verbose=args.verbose,
            fresh=args.fresh
        )

    elif args.module == pagerank_module:
        print("pagerank in main ...")
        # Setup LDA first
        lda_results = lda.setup(
            lda_no_topics=args.lda_no_topics,
            bertopic_no_topics=0,   # Force flat LDA only
            verbose=getattr(args, 'verbose', False),
            fresh=args.fresh
        )
        log_msg = f"[PAGERANK SETUP] mode: {args.pr_mode}, LDA results for lda_no_topics={args.lda_no_topics}"

        # Setup PageRank
        pr = pagerank.setup(
            log_msg=log_msg,
            lda_results=lda_results,
            topic_model=getattr(args, 'topic_model', TopicModelModes.LDA),
            no_topics=args.no_topics,
            pr_mode=args.pr_mode,
            bertopic_no_topics=0,   # Force flat LDA only
            fresh=args.fresh,
            weights=getattr(args, 'weights', '0.25,0.25,0.25,0.25')
        )

        print(f"PageRank computation completed for mode: {args.pr_mode}")

    elif args.module == evaluation_module:
        lda_no_topics_val = getattr(args, 'lda_no_topics', 200)
        bertopic_no_topics_val = getattr(args, 'bertopic_no_topics', 0)
        pr_mode_val = PRModes.argparse(args.pr_mode)

        # Setup LDA
        lda_results = lda.setup(
            lda_no_topics=lda_no_topics_val,
            bertopic_no_topics=0,   # Force flat LDA only
            verbose=getattr(args, 'verbose', False),
            fresh=getattr(args, 'fresh', False)
        )

        log_msg = f"[EVALUATION PAGERANK SETUP] mode: {pr_mode_val}, lda_no_topics={lda_no_topics_val}"

        # Setup PageRank (load cached results)
        pr = pagerank.setup(
            log_msg=log_msg,
            lda_results=lda_results,
            topic_model=TopicModelModes.LDA,
            no_topics=args.no_topics,
            pr_mode=pr_mode_val,
            bertopic_no_topics=bertopic_no_topics_val,
            fresh=getattr(args, 'fresh', False)
        )

        # Run evaluation
        evaluation.setup(
            no_topics=args.no_topics,
            pr_mode=pr_mode_val,
            agify_results=getattr(args, 'agify_results', True),
            fresh=getattr(args, 'fresh', False),
            lda_no_topics=lda_no_topics_val,
            bertopic_no_topics=bertopic_no_topics_val
        )

    elif args.module == query_module:
        # Setup LDA
        lda_results = lda.setup(
            lda_no_topics=args.lda_no_topics,
            bertopic_no_topics=0,   # Force flat LDA only
            verbose=getattr(args, 'verbose', False),
            fresh=getattr(args, 'fresh', False)
        )

        # Setup PageRank
        pr = pagerank.setup(
            topic_model=TopicModelModes.LDA,
            no_topics=args.no_topics,
            pr_mode=getattr(args, 'pr_mode', PRModes.NORMAL),
            fresh=getattr(args, 'fresh', False)
        )

        # Create QueryManager
        qm = query.QueryManager(
            page_rank=pr,
            lda_results=lda_results,
            n_components=getattr(args, 'n_components', 100)
        )

        # Perform query
        result = qm.query(
            query=args.query,
            agify_results=args.agify_results,
            count=args.count,
            standard=args.standard,
            method=args.method
        )

        # Print results
        print(f"\n{'=' * 80}")
        print(f"Query: '{result.query}'")
        print(f"Method: {args.method}")
        print(f"Found {len(result.papers)} papers")
        print(f"{'=' * 80}\n")

        for i, paper in enumerate(result.papers, 1):
            print(f"{i}. {paper.title} ({paper.year})")
            authors = [a.formatted_name for a in paper.authors]
            print(f"   Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")
            print(f"   Venue: {paper.venue}")
            print()

    elif args.module == baselines_module:
        baselines.setup_all_baselines(fresh=getattr(args, 'fresh', False))
        print(f"Baseline methods setup completed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
