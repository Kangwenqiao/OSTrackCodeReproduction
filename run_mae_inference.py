#!/usr/bin/env python
"""
使用MAE预训练模型在OTB数据集上进行推理并生成性能报告

注意: MAE模型只有backbone权重，tracking head是随机初始化的，性能会受限
"""
import os
import sys
import argparse
import torch
import csv
from datetime import datetime

# 添加项目路径
prj_path = os.path.dirname(os.path.abspath(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import print_results, check_and_load_precomputed_results, merge_multiple_runs, get_auc_curve, get_prec_curve
from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file


def create_mae_checkpoint(mae_path, output_path):
    """
    创建一个可用于推理的检查点（backbone用MAE权重，head随机初始化）
    """
    print(f"\n{'='*80}")
    print("准备MAE预训练模型用于推理")
    print(f"{'='*80}")
    print(f"MAE模型路径: {mae_path}")
    print(f"输出检查点路径: {output_path}")
    
    # 加载配置
    yaml_file = os.path.join(prj_path, 'experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml')
    update_config_from_file(yaml_file)
    
    # 构建模型
    print("\n构建OSTrack模型...")
    model = build_ostrack(cfg, training=False)
    
    # 加载MAE权重到backbone
    print(f"加载MAE预训练权重...")
    mae_checkpoint = torch.load(mae_path, map_location='cpu')
    
    if 'model' in mae_checkpoint:
        mae_state_dict = mae_checkpoint['model']
    else:
        mae_state_dict = mae_checkpoint
    
    # 加载backbone权重
    missing_keys, unexpected_keys = model.backbone.load_state_dict(mae_state_dict, strict=False)
    print(f"  - 缺失的键: {len(missing_keys)}")
    print(f"  - 意外的键: {len(unexpected_keys)}")
    print(f"  - Backbone权重已加载")
    print(f"  - Tracking head使用随机初始化（性能会受限）")
    
    # 创建检查点目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为OSTrack格式的检查点
    checkpoint = {
        'net': model.state_dict(),
        'epoch': 0,
        'note': 'MAE pretrained backbone with random initialized head - for inference only'
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n检查点已保存: {output_path}")
    print(f"{'='*80}\n")
    
    return output_path


def run_inference(tracker_name, tracker_param, dataset_name, threads, num_gpus):
    """
    在数据集上运行推理
    """
    print(f"\n{'='*80}")
    print("开始推理")
    print(f"{'='*80}")
    print(f"跟踪器: {tracker_name}")
    print(f"配置: {tracker_param}")
    print(f"数据集: {dataset_name}")
    print(f"线程数: {threads}")
    print(f"GPU数: {num_gpus}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    print(f"加载{dataset_name}数据集...")
    dataset = get_dataset(dataset_name)
    print(f"找到 {len(dataset)} 个视频序列\n")
    
    # 创建跟踪器
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id=None)]
    
    # 运行推理
    print("开始推理...\n")
    run_dataset(dataset, trackers, debug=0, threads=threads, num_gpus=num_gpus)
    
    print(f"\n{'='*80}")
    print("推理完成!")
    print(f"{'='*80}\n")


def analyze_results(tracker_name, tracker_param, dataset_name, display_name):
    """
    分析结果并生成性能报告和CSV文件
    """
    print(f"\n{'='*80}")
    print("生成性能报告")
    print(f"{'='*80}\n")
    
    # 检查结果是否存在
    # OSTrack的结果存储在 tracker_name/tracker_param/ 目录下，不包含dataset_name子目录
    results_dir = os.path.join(
        prj_path, 'output', 'test', 'tracking_results',
        tracker_name, tracker_param
    )
    
    if not os.path.exists(results_dir):
        print(f"错误: 找不到结果文件 at: {results_dir}")
        return False
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt') and not f.endswith('_time.txt')]
    print(f"结果目录: {results_dir}")
    print(f"结果文件数: {len(result_files)}\n")
    
    # 创建跟踪器列表
    trackers = []
    trackers.extend(trackerlist(
        name=tracker_name,
        parameter_name=tracker_param,
        dataset_name=dataset_name,
        run_ids=None,
        display_name=display_name
    ))
    
    # 加载数据集
    dataset = get_dataset(dataset_name)
    
    # 加载预计算的结果用于生成CSV
    eval_data = check_and_load_precomputed_results(trackers, dataset, dataset_name)
    eval_data = merge_multiple_runs(eval_data)
    
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    
    # 计算各项指标
    scores = {}
    threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
    scores['AUC'] = auc
    scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
    scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]
    
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
    scores['Precision'] = prec_score
    
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
    scores['Norm Precision'] = norm_prec_score
    
    # 打印性能指标
    print("="*80)
    print("性能指标")
    print("="*80)
    print_results(trackers, dataset, dataset_name, 
                 merge_results=True, 
                 plot_types=('success', 'norm_prec', 'prec'))
    
    # 生成CSV报告
    csv_path = generate_csv_report(
        tracker_name, tracker_param, dataset_name, display_name,
        scores, len(result_files), valid_sequence.long().sum().item()
    )
    
    print("\n" + "="*80)
    print("报告生成完成")
    print("="*80)
    print(f"\n结果已保存到: {results_dir}")
    print(f"CSV报告已保存到: {csv_path}")
    print(f"{'='*80}\n")
    
    return True


def generate_csv_report(tracker_name, tracker_param, dataset_name, display_name, 
                       scores, total_sequences, valid_sequences):
    """
    生成CSV格式的性能报告
    """
    # 创建output目录
    output_dir = os.path.join(prj_path, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成CSV文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'performance_report_{tracker_name}_{tracker_param}_{dataset_name}_{timestamp}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    
    # 写入CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入标题信息
        writer.writerow(['OSTrack Performance Report'])
        writer.writerow(['Generated at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        
        # 写入配置信息
        writer.writerow(['Configuration'])
        writer.writerow(['Tracker Name', tracker_name])
        writer.writerow(['Tracker Parameter', tracker_param])
        writer.writerow(['Display Name', display_name])
        writer.writerow(['Dataset', dataset_name])
        writer.writerow(['Total Sequences', total_sequences])
        writer.writerow(['Valid Sequences', valid_sequences])
        writer.writerow([])
        
        # 写入性能指标
        writer.writerow(['Performance Metrics'])
        writer.writerow(['Metric', 'Value (%)'])
        
        for metric_name, metric_value in scores.items():
            # 提取数值（tensor转为float）
            if torch.is_tensor(metric_value):
                value = metric_value.item() if metric_value.numel() == 1 else metric_value[0].item()
            else:
                value = metric_value
            writer.writerow([metric_name, f'{value:.2f}'])
        
        writer.writerow([])
        
        # 写入注释
        writer.writerow(['Notes'])
        writer.writerow(['This model uses MAE pretrained backbone with randomly initialized tracking head'])
        writer.writerow(['Performance is limited compared to fully trained models'])
        writer.writerow(['For better performance, use fully trained OSTrack checkpoints'])
    
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='使用MAE预训练模型在OTB数据集上推理并生成性能报告'
    )
    parser.add_argument('--mae_model', type=str, 
                       default='pretrained_models/mae_pretrain_vit_base.pth',
                       help='MAE预训练模型路径')
    parser.add_argument('--dataset', type=str, default='otb',
                       help='数据集名称 (otb, lasot, got10k等)')
    parser.add_argument('--threads', type=int, default=4,
                       help='并行线程数')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='GPU数量')
    parser.add_argument('--skip_inference', action='store_true',
                       help='跳过推理，仅生成报告（如果结果已存在）')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OSTrack MAE预训练模型推理和性能评估")
    print("="*80)
    print("\n警告: MAE模型只包含backbone权重，tracking head是随机初始化的")
    print("      推理性能会明显低于完全训练的模型!")
    print("      如需更好性能，请下载完整训练的OSTrack检查点\n")
    print("="*80 + "\n")
    
    # 设置路径
    mae_path = os.path.join(prj_path, args.mae_model)
    tracker_name = 'ostrack'
    tracker_param = 'vitb_256_mae_ce_32x4_ep300'
    checkpoint_dir = os.path.join(
        prj_path, 'output', 'checkpoints', 'train', 
        tracker_name, tracker_param
    )
    checkpoint_path = os.path.join(checkpoint_dir, 'OSTrack_ep0300.pth.tar')
    
    # 检查MAE模型是否存在
    if not os.path.exists(mae_path):
        print(f"错误: 找不到MAE预训练模型: {mae_path}")
        sys.exit(1)
    
    # 创建或检查检查点
    if not os.path.exists(checkpoint_path) or not args.skip_inference:
        create_mae_checkpoint(mae_path, checkpoint_path)
    
    # 运行推理
    if not args.skip_inference:
        run_inference(
            tracker_name=tracker_name,
            tracker_param=tracker_param,
            dataset_name=args.dataset,
            threads=args.threads,
            num_gpus=args.num_gpus
        )
    
    # 生成性能报告
    success = analyze_results(
        tracker_name=tracker_name,
        tracker_param=tracker_param,
        dataset_name=args.dataset,
        display_name='OSTrack-MAE'
    )
    
    if success:
        print("\n" + "="*80)
        print("全部完成!")
        print("="*80)
        print("\n结果概要:")
        print(f"  - 模型: MAE预训练 (backbone only)")
        print(f"  - 数据集: {args.dataset}")
        print(f"  - 结果目录: output/test/tracking_results/{tracker_name}/{tracker_param}/{args.dataset}/")
        print("="*80 + "\n")
    else:
        print("\n错误: 性能报告生成失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
