import pandas as pd

def main():
    
    csv_path = 'data/index.csv'
    df = pd.read_csv(csv_path)
    
    df = df.rename(columns={
        'training or validation': 'split',
        'nodularity': 'nodularity',
        'nodule size': 'size'
    })
    
    # Tally nodularity by train/validation
    nodularity_counts = (
        df
        .groupby(['split', 'nodularity'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    
    # Tally nodule size by train/validation
    size_counts = (
        df
        .groupby(['split', 'size'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    
    # Print results
    print("=== Nodularity Counts ===")
    print(nodularity_counts.to_string())
    
    print("\n=== Nodule Size Counts ===")
    print(size_counts.to_string())

if __name__ == '__main__':
    main()
