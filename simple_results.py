#!/usr/bin/env python3
import argparse
import os

def get_results(dataset):
    """Lấy kết quả từ một dataset"""
    file_path = f"code/result/{dataset}/training_statistics.csv"
    
    if not os.path.exists(file_path):
        return None
    
    # Đọc file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Lấy metadata (dòng cuối)
    metadata_line = lines[-1].strip()
    metadata = metadata_line.split(',')
    
    test_acc = float(metadata[2])
    test_f1 = float(metadata[3])
    
    # Parse training data để tìm best validation
    best_val_f1 = 0
    best_val_acc = 0
    best_train_f1 = 0
    best_train_acc = 0
    best_step = 0
    
    for line in lines[1:-2]:  # Bỏ header và 2 dòng cuối
        if not line.strip():
            continue
        
        cols = line.strip().split(',')
        try:
            step = int(cols[0])
            acc_val = float(cols[1])
            f1_val = float(cols[2])
            acc_train = float(cols[3])
            f1_train = float(cols[4])
            
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_val_acc = acc_val
                best_train_f1 = f1_train
                best_train_acc = acc_train
                best_step = step
        except:
            continue
    
    return {
        'dataset': dataset,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'train_acc': best_train_acc,
        'train_f1': best_train_f1,
        'best_step': best_step
    }

def show_single_dataset(result):
    """Hiển thị kết quả chi tiết cho 1 dataset"""
    print(f"🏆 KẾT QUẢ {result['dataset'].upper()} EXPERIMENT")
    print("="*50)
    
    print(f"\n📊 BẢNG KẾT QUẢ CHI TIẾT:")
    print("+"+ "-"*48 + "+")
    print(f"| {'Metric':<20} | {'Value':<8} | {'Percentage':<12} |")
    print("+"+ "-"*48 + "+")
    print(f"| {'Test Accuracy':<20} | {result['test_acc']:<8.4f} | {result['test_acc']*100:<11.2f}% |")
    print(f"| {'Test F1-Score':<20} | {result['test_f1']:<8.4f} | {result['test_f1']*100:<11.2f}% |")
    print("+"+ "-"*48 + "+")
    print(f"| {'Val Accuracy':<20} | {result['val_acc']:<8.4f} | {result['val_acc']*100:<11.2f}% |")
    print(f"| {'Val F1-Score':<20} | {result['val_f1']:<8.4f} | {result['val_f1']*100:<11.2f}% |")
    print("+"+ "-"*48 + "+")
    print(f"| {'Train Accuracy':<20} | {result['train_acc']:<8.4f} | {result['train_acc']*100:<11.2f}% |")
    print(f"| {'Train F1-Score':<20} | {result['train_f1']:<8.4f} | {result['train_f1']*100:<11.2f}% |")
    print("+"+ "-"*48 + "+")
    print(f"Best validation at step: {result['best_step']}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")
    print(f"acc_val={result['val_acc']:.4f}, f1_val={result['val_f1']:.4f}")
    print(f"acc_train={result['train_acc']:.4f}, f1_train={result['train_f1']:.4f}")

def show_multiple_datasets(results):
    """Hiển thị bảng so sánh nhiều dataset"""
    print(f"🏆 SO SÁNH KẾT QUẢ {len(results)} DATASETS")
    print("="*80)
    
    # Header bảng
    header = f"| {'Dataset':<12} | {'Test ACC':<9} | {'Test F1':<8} | {'Val ACC':<8} | {'Val F1':<7} | {'Train ACC':<9} | {'Train F1':<8} |"
    separator = "+" + "-"*(len(header)-2) + "+"
    
    print(separator)
    print(header)
    print(separator)
    
    # Dữ liệu từng dataset
    for result in results:
        row = (f"| {result['dataset']:<12} | "
               f"{result['test_acc']:<9.4f} | "
               f"{result['test_f1']:<8.4f} | "
               f"{result['val_acc']:<8.4f} | "
               f"{result['val_f1']:<7.4f} | "
               f"{result['train_acc']:<9.4f} | "
               f"{result['train_f1']:<8.4f} |")
        print(row)
    
    print(separator)
    
    # Tìm dataset tốt nhất
    best_test_f1 = max(results, key=lambda x: x['test_f1'])
    best_val_f1 = max(results, key=lambda x: x['val_f1'])
    
    print(f"\n� BEST RESULTS:")
    print(f"   Best Test F1:  {best_test_f1['dataset']} ({best_test_f1['test_f1']:.4f})")
    print(f"   Best Val F1:   {best_val_f1['dataset']} ({best_val_f1['val_f1']:.4f})")
    
    # Summary chi tiết từng dataset
    print(f"\n📋 DETAILED SUMMARY:")
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")
        print(f"  acc_val={result['val_acc']:.4f}, f1_val={result['val_f1']:.4f}")
        print(f"  acc_train={result['train_acc']:.4f}, f1_train={result['train_f1']:.4f}")
        print(f"  best_step={result['best_step']}")

def main():
    parser = argparse.ArgumentParser(description='Hiển thị kết quả experiment')
    parser.add_argument('--dataset', nargs='*', default=['uit-vsfc'], 
                       help='Dataset name(s) (default: uit-vsfc). Có thể nhập nhiều: --dataset uit-vsfc yahoo')
    
    args = parser.parse_args()
    
    # Lấy kết quả từ tất cả datasets
    results = []
    missing_datasets = []
    
    for dataset in args.dataset:
        result = get_results(dataset)
        if result:
            results.append(result)
        else:
            missing_datasets.append(dataset)
    
    # Thông báo dataset không tìm thấy
    if missing_datasets:
        print(f"❌ Không tìm thấy kết quả cho: {', '.join(missing_datasets)}")
        print("📁 Available datasets:")
        if os.path.exists("code/result"):
            for item in os.listdir("code/result"):
                if os.path.isdir(f"code/result/{item}"):
                    print(f"   - {item}")
        print()
    
    # Hiển thị kết quả
    if not results:
        print("❌ Không có dataset nào có kết quả để hiển thị!")
        return
    
    if len(results) == 1:
        # Hiển thị chi tiết cho 1 dataset
        show_single_dataset(results[0])
    else:
        # Hiển thị bảng so sánh cho nhiều dataset
        show_multiple_datasets(results)

if __name__ == "__main__":
    main()
