#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese WordNet Synonym Dictionary Builder
Đọc tất cả file CSV trong vi-wordnet và tạo file JSON tổng hợp
"""

import os
import csv
import json
import glob
from collections import defaultdict
import re

def clean_word(word):
    """Làm sạch từ: loại bỏ khoảng trắng thừa, ký tự đặc biệt"""
    if not word:
        return ""
    
    # Loại bỏ khoảng trắng đầu cuối
    word = word.strip()
    
    # Loại bỏ các ký tự đặc biệt thừa
    word = re.sub(r'[,\s]+$', '', word)  # Loại bỏ dấu phẩy cuối
    word = re.sub(r'^[,\s]+', '', word)  # Loại bỏ dấu phẩy đầu
    
    return word

def parse_synonym_line(line):
    """
    Parse một dòng synonym từ CSV
    Return: list các từ đồng nghĩa
    """
    if not line.strip():
        return []
    
    # Split bằng dấu phẩy
    words = [clean_word(word) for word in line.split(',')]
    
    # Loại bỏ từ rỗng
    words = [word for word in words if word and len(word) > 1]
    
    return words

def build_synonym_dict():
    """
    Đọc tất cả file CSV và xây dựng từ điển đồng nghĩa
    """
    synonym_dict = defaultdict(set)
    category_stats = defaultdict(int)
    
    # Tìm tất cả file CSV
    csv_files = glob.glob("*.csv")
    
    print(f"🔍 Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"📖 Processing: {csv_file}")
        category = csv_file.replace('.csv', '')
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                # Parse synonyms từ dòng
                synonyms = parse_synonym_line(line)
                
                if len(synonyms) >= 2:  # Cần ít nhất 2 từ để tạo quan hệ đồng nghĩa
                    # Tạo quan hệ đồng nghĩa cho tất cả các từ
                    for word in synonyms:
                        # Thêm tất cả từ khác làm đồng nghĩa của word này
                        other_words = [w for w in synonyms if w != word]
                        synonym_dict[word].update(other_words)
                        
                    category_stats[category] += 1
                    
        except Exception as e:
            print(f"  ❌ Error processing {csv_file}: {e}")
            continue
    
    # Convert set thành list để JSON serializable
    final_dict = {}
    for word, synonyms in synonym_dict.items():
        if synonyms:  # Chỉ giữ từ có đồng nghĩa
            final_dict[word] = list(synonyms)
    
    return final_dict, category_stats

def save_synonym_files(synonym_dict, category_stats):
    """
    Lưu từ điển đồng nghĩa ra các file khác nhau
    """
    
    # 1. Lưu file JSON chính
    print("💾 Saving vi_synonyms.json...")
    with open('vi_synonyms.json', 'w', encoding='utf-8') as f:
        json.dump(synonym_dict, f, ensure_ascii=False, indent=2)
    
    # 2. Lưu file TXT dễ đọc cho con người
    print("💾 Saving vi_synonyms.txt...")
    with open('vi_synonyms.txt', 'w', encoding='utf-8') as f:
        f.write("# Vietnamese Synonym Dictionary\n")
        f.write(f"# Total words: {len(synonym_dict)}\n")
        f.write(f"# Generated from vi-wordnet CSV files\n\n")
        
        for word in sorted(synonym_dict.keys()):
            synonyms = synonym_dict[word]
            f.write(f"{word}: {', '.join(synonyms)}\n")
    
    # 3. Lưu file Python dict để import trực tiếp
    print("💾 Saving vi_synonyms.py...")
    with open('vi_synonyms.py', 'w', encoding='utf-8') as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""\n')
        f.write('Vietnamese Synonym Dictionary\n')
        f.write('Auto-generated from vi-wordnet CSV files\n')
        f.write('"""\n\n')
        f.write('VI_SYNONYMS = ')
        
        # Pretty print dictionary
        import pprint
        f.write(pprint.pformat(synonym_dict, width=100))
    
    # 4. Lưu statistics
    print("💾 Saving statistics...")
    stats = {
        'total_words': len(synonym_dict),
        'total_synonym_pairs': sum(len(syns) for syns in synonym_dict.values()),
        'category_breakdown': dict(category_stats),
        'average_synonyms_per_word': sum(len(syns) for syns in synonym_dict.values()) / len(synonym_dict) if synonym_dict else 0
    }
    
    with open('vi_synonyms_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats

def main():
    """Main function"""
    print("🚀 Vietnamese WordNet Synonym Dictionary Builder")
    print("="*60)
    
    # Đã ở trong thư mục vi-wordnet rồi
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Build synonym dictionary
    print("🔨 Building synonym dictionary...")
    synonym_dict, category_stats = build_synonym_dict()
    
    # Save files
    stats = save_synonym_files(synonym_dict, category_stats)
    
    # Print results
    print("\n📊 RESULTS:")
    print(f"  Total unique words: {stats['total_words']:,}")
    print(f"  Total synonym relationships: {stats['total_synonym_pairs']:,}")
    print(f"  Average synonyms per word: {stats['average_synonyms_per_word']:.1f}")
    
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, count in sorted(category_stats.items()):
        print(f"  {category}: {count:,} synonym groups")
    
    print(f"\n📁 FILES CREATED:")
    print(f"  📄 vi_synonyms.json - Main dictionary (for code)")
    print(f"  📄 vi_synonyms.txt - Human readable format")
    print(f"  📄 vi_synonyms.py - Python importable")
    print(f"  📄 vi_synonyms_stats.json - Statistics")
    
    print(f"\n✅ Synonym dictionary building complete!")
    
    # Example usage
    print(f"\n💡 EXAMPLE USAGE:")
    example_words = list(synonym_dict.keys())[:3]
    for word in example_words:
        synonyms = synonym_dict[word][:5]  # First 5 synonyms
        print(f"  '{word}' → {synonyms}")

if __name__ == "__main__":
    main()
