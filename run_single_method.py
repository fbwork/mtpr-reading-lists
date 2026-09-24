"""
اسکریپت برای اجرای یک روش PageRank خاص

استفاده:
    python run_single_method.py [METHOD_NAME]
    
مثال:
    python run_single_method.py hybrid
    python run_single_method.py gat
    python run_single_method.py flow_matching
"""

import sys
import subprocess
from pathlib import Path

# پارامترهای پیش‌فرض
NO_TOPICS = 5
LDA_NO_TOPICS = 200
BERTOPIC_NO_TOPICS = 0
TOPIC_MODEL = 'lda'

# لیست تمام روش‌های موجود
ALL_METHODS = [
    # Basic
    'normal', 'personalized', 'themed',
    # Topic-Modified
    'modified', 'modified_bias', 'modified_transition',
    # Degree Centrality
    'd_modified', 'd_modified_bias', 'd_modified_transition',
    # Closeness Centrality
    'cl_modified', 'cl_modified_bias', 'cl_modified_transition',
    # Betweenness Centrality
    'bw_modified', 'bw_modified_bias', 'bw_modified_transition',
    # Katz Centrality
    'kz_modified', 'kz_modified_bias', 'kz_modified_transition',
    # Hybrid
    'hybrid',
    # GAT
    'gat', 'gat_centrality', 'gat_degree', 'gat_closeness', 
    'gat_betweenness', 'gat_katz', 'gat_topic_aware',
    'gat_mtpr', 'gat_mtpr_centrality', 'gat_mtpr_embedding',
    # Flow-Based
    'flow_matching'
]

# توضیحات روش‌ها
METHOD_DESCRIPTIONS = {
    'normal': 'PageRank استاندارد با توزیع یکنواخت',
    'personalized': 'PageRank با personalization vector مبتنی بر topic',
    'themed': 'PageRank با transition probabilities مبتنی بر topic',
    'modified': 'PageRank با Topic + Author scores',
    'modified_bias': 'PageRank با Topic + Author scores (فقط bias)',
    'modified_transition': 'PageRank با Topic + Author scores (فقط transition)',
    'd_modified': 'PageRank با Degree centrality',
    'd_modified_bias': 'PageRank با Degree centrality (فقط bias)',
    'd_modified_transition': 'PageRank با Degree centrality (فقط transition)',
    'cl_modified': 'PageRank با Closeness centrality',
    'cl_modified_bias': 'PageRank با Closeness centrality (فقط bias)',
    'cl_modified_transition': 'PageRank با Closeness centrality (فقط transition)',
    'bw_modified': 'PageRank با Betweenness centrality',
    'bw_modified_bias': 'PageRank با Betweenness centrality (فقط bias)',
    'bw_modified_transition': 'PageRank با Betweenness centrality (فقط transition)',
    'kz_modified': 'PageRank با Katz centrality',
    'kz_modified_bias': 'PageRank با Katz centrality (فقط bias)',
    'kz_modified_transition': 'PageRank با Katz centrality (فقط transition)',
    'hybrid': 'ترکیب وزندار تمام centrality measures',
    'gat': 'Graph Attention Network استاندارد (GPU)',
    'gat_centrality': 'GAT با ویژگی‌های centrality (GPU)',
    'gat_degree': 'GAT با degree centrality weighting (GPU)',
    'gat_closeness': 'GAT با closeness centrality weighting (GPU)',
    'gat_betweenness': 'GAT با betweenness centrality weighting (GPU)',
    'gat_katz': 'GAT با Katz centrality weighting (GPU)',
    'gat_topic_aware': 'GAT با topic-specific attention heads (GPU)',
    'gat_mtpr': 'GAT + Multi-Topic PageRank fusion (GPU)',
    'gat_mtpr_centrality': 'GAT + MTPR + Centrality features (GPU)',
    'gat_mtpr_embedding': 'GAT + MTPR with learned embeddings (GPU)',
    'flow_matching': 'Flow Matching برای citation path prediction (GPU)',
}


def print_available_methods():
    """چاپ لیست تمام روش‌های موجود"""
    print("\n" + "="*80)
    print("روش‌های PageRank موجود:")
    print("="*80)
    
    categories = {
        'Basic PageRank (3)': ['normal', 'personalized', 'themed'],
        'Topic-Modified (3)': ['modified', 'modified_bias', 'modified_transition'],
        'Degree Centrality (3)': ['d_modified', 'd_modified_bias', 'd_modified_transition'],
        'Closeness Centrality (3)': ['cl_modified', 'cl_modified_bias', 'cl_modified_transition'],
        'Betweenness Centrality (3)': ['bw_modified', 'bw_modified_bias', 'bw_modified_transition'],
        'Katz Centrality (3)': ['kz_modified', 'kz_modified_bias', 'kz_modified_transition'],
        'Hybrid (1)': ['hybrid'],
        'GAT Methods (10)': ['gat', 'gat_centrality', 'gat_degree', 'gat_closeness', 
                            'gat_betweenness', 'gat_katz', 'gat_topic_aware',
                            'gat_mtpr', 'gat_mtpr_centrality', 'gat_mtpr_embedding'],
        'Flow-Based (1)': ['flow_matching'],
    }
    
    for category, methods in categories.items():
        print(f"\n{category}:")
        for method in methods:
            desc = METHOD_DESCRIPTIONS.get(method, '')
            print(f"  - {method:30} : {desc}")
    
    print("\n" + "="*80)


def run_method(method_name, fresh=False):
    """اجرای یک روش PageRank خاص"""
    
    if method_name not in ALL_METHODS:
        print(f"❌ خطا: روش '{method_name}' معتبر نیست!")
        print_available_methods()
        return False
    
    print("\n" + "="*80)
    print(f"اجرای روش: {method_name}")
    print(f"توضیح: {METHOD_DESCRIPTIONS.get(method_name, '')}")
    print("="*80)
    
    # بررسی GPU برای روش‌های GAT
    if method_name.startswith('gat') or method_name == 'flow_matching':
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU در دسترس: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  هشدار: GPU در دسترس نیست، اجرا روی CPU کند خواهد بود")
    
    # ساخت دستور
    cmd = [
        'python', 'main.py', 'pagerank',
        '--no-topics', str(NO_TOPICS),
        '--pr-mode', method_name,
        '--lda-no-topics', str(LDA_NO_TOPICS),
        '--bertopic-no-topics', str(BERTOPIC_NO_TOPICS),
        '--topic-model', TOPIC_MODEL
    ]
    
    if fresh:
        cmd.append('--fresh')
    
    # اضافه کردن weights برای hybrid
    if method_name == 'hybrid':
        cmd.extend(['--weights', '0.25,0.25,0.25,0.25'])
    
    print(f"\nدستور اجرا: {' '.join(cmd)}\n")
    
    # اجرای دستور
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ روش '{method_name}' با موفقیت اجرا شد!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ خطا در اجرای روش '{method_name}': {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  اجرا توسط کاربر متوقف شد.")
        return False


def main():
    """تابع اصلی"""
    if len(sys.argv) < 2:
        print("استفاده: python run_single_method.py [METHOD_NAME] [--fresh]")
        print("\nمثال:")
        print("  python run_single_method.py hybrid")
        print("  python run_single_method.py gat --fresh")
        print("  python run_single_method.py list  # برای نمایش لیست روش‌ها")
        print_available_methods()
        return
    
    method_name = sys.argv[1].lower()
    fresh = '--fresh' in sys.argv
    
    if method_name == 'list':
        print_available_methods()
        return
    
    success = run_method(method_name, fresh=fresh)
    
    if success:
        print("\n" + "="*80)
        print("✅ اجرا کامل شد!")
        print(f"نتایج در: components/output/pr/{NO_TOPICS}/{NO_TOPICS}/pageranks_{method_name}.bin")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ اجرا با خطا مواجه شد!")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()





