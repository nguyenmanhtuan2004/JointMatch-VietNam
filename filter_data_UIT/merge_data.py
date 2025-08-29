#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để ghép topics.txt và sents.txt thành dataset
Chuyển đổi labels: 0->1, 1->2, 2->3, 3->4
"""

import pandas as pd
import os

def merge_topics_sents(train_dir):
    """
    Ghép file topics.txt và sents.txt thành dataset
    
    Args:
        train_dir: Đường dẫn đến thư mục chứa topics.txt và sents.txt
    
    Returns:
        DataFrame với cột 'label' và 'content'
    """
    
    topics_file = os.path.join(train_dir, 'topics.txt')
    sents_file = os.path.join(train_dir, 'sents.txt')
    
    print(f"📂 Đọc file topics: {topics_file}")
    print(f"📂 Đọc file sents: {sents_file}")
    
    # Đọc file topics.txt
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = [int(line.strip()) for line in f.readlines()]
    
    # Đọc file sents.txt  
    with open(sents_file, 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f.readlines()]
    
    print(f"📊 Số lượng topics: {len(topics)}")
    print(f"📊 Số lượng sentences: {len(sents)}")
    
    # Kiểm tra độ dài khớp nhau
    if len(topics) != len(sents):
        raise ValueError(f"Số lượng topics ({len(topics)}) và sentences ({len(sents)}) không khớp!")
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'label': topics,
        'content': sents
    })
    
    print(f"✅ Đã tạo DataFrame với {len(df)} dòng")
    
    # Hiển thị distribution ban đầu
    print("\n📈 Label distribution (before mapping):")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")
    
    # Chuyển đổi labels: 0->1, 1->2, 2->3, 3->4
    print("\n🔄 Chuyển đổi labels: 0->1, 1->2, 2->3, 3->4")
    df['label'] = df['label'] + 1
    
    # Hiển thị distribution sau khi chuyển đổi
    print("\n📈 Label distribution (after mapping):")
    label_counts_new = df['label'].value_counts().sort_index()
    for label, count in label_counts_new.items():
        print(f"  Label {label}: {count} samples")
    
    # Lấy tối đa N mẫu mỗi lớp
    print("\n✂️ Lấy N mẫu mỗi lớp...")
    samples_per_class = 50
    
    # Tạo list để chứa dữ liệu được lọc
    filtered_dfs = []
    
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        if len(label_df) > samples_per_class:
            
            sampled_df = label_df.sample(n=samples_per_class, random_state=42)
        else:
            
            sampled_df = label_df
        
        filtered_dfs.append(sampled_df)
        print(f"  Label {label}: {len(sampled_df)}/{len(label_df)} samples")
    
    # Ghép lại các DataFrame
    df_filtered = pd.concat(filtered_dfs, ignore_index=True)
    
    # Shuffle lại dữ liệu
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Dataset sau khi lọc: {len(df_filtered)} samples")
    
    # Hiển thị distribution cuối cùng
    print("\n📈 Final label distribution:")
    final_counts = df_filtered['label'].value_counts().sort_index()
    for label, count in final_counts.items():
        print(f"  Label {label}: {count} samples")
    
    return df_filtered

def main():
    """Main function"""
    
    # Đường dẫn thư mục train
    dir_name = "dev"
    
    try:
        # Ghép dữ liệu
        df = merge_topics_sents(dir_name)
        
        # Hiển thị một vài ví dụ
        print("\n📋 VÍ DỤ DỮ LIỆU:")
        print("="*80)
        for i in range(min(5, len(df))):
            print(f"\n{i+1}. Label {df.iloc[i]['label']}: {df.iloc[i]['content']}")
        print("="*80)
        
        # Lưu thành CSV
        output_file = f"{dir_name}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Đã lưu dataset vào: {output_file}")
        
        # Thống kê cuối cùng
        print(f"\n📊 THỐNG KÊ CUỐI CÙNG:")
        print(f"  Tổng số mẫu: {len(df)}")
        print(f"  Số lượng labels: {df['label'].nunique()}")
        print(f"  Labels range: {df['label'].min()} - {df['label'].max()}")
        print(f"  Độ dài trung bình content: {df['content'].str.len().mean():.1f} ký tự")
        print(f"  Số từ trung bình: {df['content'].str.split().str.len().mean():.1f} từ")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
