import sys
import pandas as pd
import matplotlib.pyplot as plt

def parse_input():
    data = []
    current_entry = {}
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("ANN,COLBERT:"):
            if current_entry:
                data.append(current_entry)
            current_entry = {"label": line.split(":")[1].strip()}
        elif line.startswith("Time:"):
            current_entry["time"] = float(line.split(":")[1].split()[0])
        elif line.startswith("NDCG@10:"):
            current_entry["ndcg@10"] = float(line.split(":")[1])
    if current_entry:
        data.append(current_entry)
    return pd.DataFrame(data)

def is_pareto_optimal(df, row):
    return not any((df['throughput'] >= row['throughput']) & 
                   (df['accuracy'] >= row['accuracy']) & 
                   ((df['throughput'] > row['throughput']) | 
                    (df['accuracy'] > row['accuracy'])))

def main():
    df = parse_input()
    df['throughput'] = 100 / df['time']
    df['accuracy'] = df['ndcg@10']
    
    df['pareto'] = df.apply(lambda row: is_pareto_optimal(df, row), axis=1)
    pareto_df = df[df['pareto']]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_df['accuracy'], pareto_df['throughput'], marker='o')
    
    for _, row in pareto_df.iterrows():
        plt.annotate(row['label'], (row['accuracy'], row['throughput']),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Accuracy (NDCG@10)')
    plt.ylabel('Throughput (queries/second)')
    plt.title('Accuracy vs Throughput (Pareto-optimal points)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
