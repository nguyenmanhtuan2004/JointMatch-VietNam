#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích kết quả experiment cho UIT-VSFC dataset
Đọc file CSV và hiển thị thông số F1, ACC tốt nhất cho val và train
"""

import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentAnalyzer:
    def __init__(self, experiment_path):
        self.experiment_path = Path(experiment_path)
        self.results = []
        
    def find_all_experiments(self):
        """Tìm tất cả experiment folders"""
        log_path = self.experiment_path / "log"
        if not log_path.exists():
            print(f"❌ Không tìm thấy folder log tại: {log_path}")
            return []
        
        experiment_folders = [f for f in log_path.iterdir() if f.is_dir()]
        experiment_folders.sort()
        
        print(f"📁 Tìm thấy {len(experiment_folders)} experiment folders:")
        for folder in experiment_folders:
            print(f"  - {folder.name}")
        
        return experiment_folders
    
    def read_summary_csv(self, exp_folder):
        """Đọc file summary.csv từ experiment folder"""
        summary_file = exp_folder / "summary.csv"
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                return df
            except Exception as e:
                print(f"⚠️ Lỗi đọc {summary_file}: {e}")
                return None
        return None
    
    def read_training_statistics(self, exp_folder):
        """Đọc tất cả file training_statistics.csv từ các seed folders"""
        seed_results = []
        
        # Tìm các folder seed (0, 1, 2, ...)
        seed_folders = [f for f in exp_folder.iterdir() 
                       if f.is_dir() and f.name.isdigit()]
        seed_folders.sort(key=lambda x: int(x.name))
        
        for seed_folder in seed_folders:
            stats_file = seed_folder / "training_statistics.csv"
            if stats_file.exists():
                try:
                    df = pd.read_csv(stats_file)
                    df['seed'] = int(seed_folder.name)
                    df['experiment'] = exp_folder.name
                    seed_results.append(df)
                except Exception as e:
                    print(f"⚠️ Lỗi đọc {stats_file}: {e}")
        
        return seed_results
    
    def analyze_single_experiment(self, exp_folder):
        """Phân tích một experiment"""
        print(f"\n🔍 Phân tích experiment: {exp_folder.name}")
        print("="*60)
        
        # Đọc summary
        summary_df = self.read_summary_csv(exp_folder)
        if summary_df is not None:
            print(f"📊 Summary results:")
            print(f"  Seeds: {len(summary_df)}")
            print(f"  Test ACC: {summary_df['test_acc'].mean():.4f} ± {summary_df['test_acc'].std():.4f}")
            print(f"  Test F1:  {summary_df['test_f1'].mean():.4f} ± {summary_df['test_f1'].std():.4f}")
            print(f"  Best step avg: {summary_df['best_step'].mean():.1f}")
        
        # Đọc training statistics
        training_data = self.read_training_statistics(exp_folder)
        if training_data:
            print(f"\n📈 Training statistics ({len(training_data)} seeds):")
            
            best_results = {}
            for seed_df in training_data:
                seed = seed_df['seed'].iloc[0]
                
                # Tìm best validation results
                best_val_idx = seed_df['f1_val'].idxmax()
                best_val_f1 = seed_df.loc[best_val_idx, 'f1_val']
                best_val_acc = seed_df.loc[best_val_idx, 'acc_val']
                best_val_step = seed_df.loc[best_val_idx, 'step']
                
                # Training results tại step đó
                train_f1_at_best = seed_df.loc[best_val_idx, 'f1_train']
                train_acc_at_best = seed_df.loc[best_val_idx, 'acc_train']
                
                # Best training results overall
                best_train_f1 = seed_df['f1_train'].max()
                best_train_acc = seed_df['acc_train'].max()
                
                best_results[seed] = {
                    'best_val_f1': best_val_f1,
                    'best_val_acc': best_val_acc,
                    'best_val_step': best_val_step,
                    'train_f1_at_best_val': train_f1_at_best,
                    'train_acc_at_best_val': train_acc_at_best,
                    'best_train_f1': best_train_f1,
                    'best_train_acc': best_train_acc
                }
                
                print(f"  Seed {seed}:")
                print(f"    Val  F1: {best_val_f1:.4f} | ACC: {best_val_acc:.4f} (step {best_val_step})")
                print(f"    Train F1: {train_f1_at_best:.4f} | ACC: {train_acc_at_best:.4f} (at best val)")
                print(f"    Best Train F1: {best_train_f1:.4f} | ACC: {best_train_acc:.4f}")
            
            # Tính average across seeds
            val_f1_scores = [r['best_val_f1'] for r in best_results.values()]
            val_acc_scores = [r['best_val_acc'] for r in best_results.values()]
            train_f1_scores = [r['train_f1_at_best_val'] for r in best_results.values()]
            train_acc_scores = [r['train_acc_at_best_val'] for r in best_results.values()]
            
            print(f"\n📊 TỔNG KẾT EXPERIMENT {exp_folder.name}:")
            print(f"  🎯 VALIDATION (best):")
            print(f"     F1:  {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}")
            print(f"     ACC: {np.mean(val_acc_scores):.4f} ± {np.std(val_acc_scores):.4f}")
            print(f"  🎯 TRAINING (at best val):")
            print(f"     F1:  {np.mean(train_f1_scores):.4f} ± {np.std(train_f1_scores):.4f}")
            print(f"     ACC: {np.mean(train_acc_scores):.4f} ± {np.std(train_acc_scores):.4f}")
            
            return {
                'experiment': exp_folder.name,
                'val_f1_mean': np.mean(val_f1_scores),
                'val_f1_std': np.std(val_f1_scores),
                'val_acc_mean': np.mean(val_acc_scores),
                'val_acc_std': np.std(val_acc_scores),
                'train_f1_mean': np.mean(train_f1_scores),
                'train_f1_std': np.std(train_f1_scores),
                'train_acc_mean': np.mean(train_acc_scores),
                'train_acc_std': np.std(train_acc_scores),
                'seeds': len(best_results),
                'training_data': training_data
            }
        
        return None
    
    def compare_all_experiments(self):
        """So sánh tất cả experiments"""
        experiment_folders = self.find_all_experiments()
        all_results = []
        
        for exp_folder in experiment_folders:
            result = self.analyze_single_experiment(exp_folder)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("❌ Không có kết quả nào để so sánh!")
            return
        
        print(f"\n" + "="*80)
        print("🏆 SO SÁNH TẤT CẢ EXPERIMENTS")
        print("="*80)
        
        # Tạo DataFrame để dễ so sánh
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Experiment': result['experiment'],
                'Val F1': f"{result['val_f1_mean']:.4f}±{result['val_f1_std']:.4f}",
                'Val ACC': f"{result['val_acc_mean']:.4f}±{result['val_acc_std']:.4f}",
                'Train F1': f"{result['train_f1_mean']:.4f}±{result['train_f1_std']:.4f}",
                'Train ACC': f"{result['train_acc_mean']:.4f}±{result['train_acc_std']:.4f}",
                'Seeds': result['seeds']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Tìm best experiment
        best_val_f1_idx = max(range(len(all_results)), 
                             key=lambda i: all_results[i]['val_f1_mean'])
        best_exp = all_results[best_val_f1_idx]
        
        print(f"\n🥇 EXPERIMENT TỐT NHẤT (theo Val F1): {best_exp['experiment']}")
        print(f"   Val F1:  {best_exp['val_f1_mean']:.4f} ± {best_exp['val_f1_std']:.4f}")
        print(f"   Val ACC: {best_exp['val_acc_mean']:.4f} ± {best_exp['val_acc_std']:.4f}")
        
        return all_results
    
    def plot_training_curves(self, experiment_name=None):
        """Vẽ biểu đồ training curves"""
        experiment_folders = self.find_all_experiments()
        
        if experiment_name:
            experiment_folders = [f for f in experiment_folders if f.name == experiment_name]
            if not experiment_folders:
                print(f"❌ Không tìm thấy experiment: {experiment_name}")
                return
        
        for exp_folder in experiment_folders:
            training_data = self.read_training_statistics(exp_folder)
            if not training_data:
                continue
                
            plt.figure(figsize=(15, 10))
            
            # Plot F1 scores
            plt.subplot(2, 2, 1)
            for seed_df in training_data:
                seed = seed_df['seed'].iloc[0]
                plt.plot(seed_df['step'], seed_df['f1_val'], 
                        label=f'Val Seed {seed}', alpha=0.7)
                plt.plot(seed_df['step'], seed_df['f1_train'], 
                        label=f'Train Seed {seed}', alpha=0.7, linestyle='--')
            plt.title(f'F1 Score - {exp_folder.name}')
            plt.xlabel('Step')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot Accuracy
            plt.subplot(2, 2, 2)
            for seed_df in training_data:
                seed = seed_df['seed'].iloc[0]
                plt.plot(seed_df['step'], seed_df['acc_val'], 
                        label=f'Val Seed {seed}', alpha=0.7)
                plt.plot(seed_df['step'], seed_df['acc_train'], 
                        label=f'Train Seed {seed}', alpha=0.7, linestyle='--')
            plt.title(f'Accuracy - {exp_folder.name}')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Average curves
            plt.subplot(2, 2, 3)
            all_steps = training_data[0]['step'].values
            
            val_f1_avg = np.mean([df['f1_val'].values for df in training_data], axis=0)
            val_f1_std = np.std([df['f1_val'].values for df in training_data], axis=0)
            train_f1_avg = np.mean([df['f1_train'].values for df in training_data], axis=0)
            train_f1_std = np.std([df['f1_train'].values for df in training_data], axis=0)
            
            plt.plot(all_steps, val_f1_avg, label='Val F1', color='blue', linewidth=2)
            plt.fill_between(all_steps, val_f1_avg - val_f1_std, val_f1_avg + val_f1_std, 
                           alpha=0.2, color='blue')
            plt.plot(all_steps, train_f1_avg, label='Train F1', color='red', linewidth=2)
            plt.fill_between(all_steps, train_f1_avg - train_f1_std, train_f1_avg + train_f1_std, 
                           alpha=0.2, color='red')
            plt.title(f'Average F1 ± Std - {exp_folder.name}')
            plt.xlabel('Step')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            val_acc_avg = np.mean([df['acc_val'].values for df in training_data], axis=0)
            val_acc_std = np.std([df['acc_val'].values for df in training_data], axis=0)
            train_acc_avg = np.mean([df['acc_train'].values for df in training_data], axis=0)
            train_acc_std = np.std([df['acc_train'].values for df in training_data], axis=0)
            
            plt.plot(all_steps, val_acc_avg, label='Val ACC', color='blue', linewidth=2)
            plt.fill_between(all_steps, val_acc_avg - val_acc_std, val_acc_avg + val_acc_std, 
                           alpha=0.2, color='blue')
            plt.plot(all_steps, train_acc_avg, label='Train ACC', color='red', linewidth=2)
            plt.fill_between(all_steps, train_acc_avg - train_acc_std, train_acc_avg + train_acc_std, 
                           alpha=0.2, color='red')
            plt.title(f'Average ACC ± Std - {exp_folder.name}')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'experiment_analysis_{exp_folder.name}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main function"""
    print("🚀 PHÂN TÍCH KẾT QUẢ EXPERIMENT UIT-VSFC")
    print("="*50)
    
    # Đường dẫn experiment
    experiment_path = "code/experiment/uit-vsfc"
    
    if not os.path.exists(experiment_path):
        print(f"❌ Không tìm thấy folder experiment: {experiment_path}")
        return
    
    # Tạo analyzer
    analyzer = ExperimentAnalyzer(experiment_path)
    
    # Phân tích tất cả experiments
    results = analyzer.compare_all_experiments()
    
    if results:
        # Vẽ biểu đồ cho experiment tốt nhất
        best_exp = max(results, key=lambda x: x['val_f1_mean'])
        print(f"\n📊 Vẽ biểu đồ cho experiment tốt nhất: {best_exp['experiment']}")
        analyzer.plot_training_curves(best_exp['experiment'])
        
        # Lưu kết quả ra file
        comparison_data = []
        for result in results:
            comparison_data.append({
                'experiment': result['experiment'],
                'val_f1_mean': result['val_f1_mean'],
                'val_f1_std': result['val_f1_std'],
                'val_acc_mean': result['val_acc_mean'],
                'val_acc_std': result['val_acc_std'],
                'train_f1_mean': result['train_f1_mean'],
                'train_f1_std': result['train_f1_std'],
                'train_acc_mean': result['train_acc_mean'],
                'train_acc_std': result['train_acc_std'],
                'seeds': result['seeds']
            })
        
        df_results = pd.DataFrame(comparison_data)
        df_results.to_csv('experiment_comparison.csv', index=False)
        print(f"\n💾 Đã lưu kết quả so sánh vào: experiment_comparison.csv")

if __name__ == "__main__":
    main()
