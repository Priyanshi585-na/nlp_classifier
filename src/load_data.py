from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    dataset = load_dataset("ccdv/arxiv-classification")
    df = pd.DataFrame(dataset['train'])

    label_names = dataset["train"].features["label"].names

    selected_classes = ['cs.CV', 'cs.AI', 'cs.SY', 'cs.DS', 'cs.NE']
    selected_indices = [label_names.index(cls) for cls in selected_classes]

    df = df[df['label'].isin(selected_indices)]

    df1 = df.groupby("label").sample(n = 2000, random_state=42).reset_index(drop =True)
    train_df, test_df = train_test_split(df1, test_size=0.2, random_state=42)
    return train_df, test_df

