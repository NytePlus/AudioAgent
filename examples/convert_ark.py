import kaldiio
import soundfile as sf
import os
import argparse
import numpy as np

def save_wav(data_info, output_path, default_sr):
    """处理 kaldiio 读取的数据并保存为 wav"""
    # kaldiio 读取波形数据时，如果 ark 内部带有 Wave 头，会返回 (sample_rate, data)
    if isinstance(data_info, tuple) and len(data_info) == 2:
        sample_rate, data = data_info
    else:
        # 否则只返回数据，采样率使用默认值
        data = data_info
        sample_rate = default_sr

    if isinstance(data, np.ndarray) and data.ndim > 1:
        print(f"⚠️ 警告: 读取到的数据是多维矩阵 (形状: {data.shape})，不是原始波形。跳过。")
        return False

    sf.write(output_path, data, sample_rate)
    return True

def extract_single(ark_offset, output_path, default_sr):
    """提取单条带有 offset 的 ark 数据"""
    print(f"正在提取: {ark_offset}")
    try:
        # kaldiio.load_mat 原生支持 "path/to/file.ark:offset" 格式
        data_info = kaldiio.load_mat(ark_offset)
        
        # 如果没有指定完整输出文件名，则自动生成
        if os.path.isdir(output_path) or not output_path.endswith('.wav'):
            os.makedirs(output_path, exist_ok=True)
            # 使用 ark 文件名和偏移量作为默认 wav 文件名
            base_name = os.path.basename(ark_offset).replace(':', '_')
            output_path = os.path.join(output_path, f"{base_name}.wav")
            
        if save_wav(data_info, output_path, default_sr):
            print(f"✅ 成功保存至: {output_path}")
            
    except Exception as e:
        print(f"❌ 提取失败: {e}")

def extract_batch_scp(scp_file, output_dir, default_sr):
    """通过读取 .scp 文件批量提取"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在读取 SCP 文件: {scp_file} ...")
    
    count = 0
    try:
        with open(scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    ark_offset = parts[1]
                    
                    try:
                        data_info = kaldiio.load_mat(ark_offset)
                        out_path = os.path.join(output_dir, f"{utt_id}.wav")
                        if save_wav(data_info, out_path, default_sr):
                            count += 1
                    except Exception as e:
                        print(f"❌ 解析 {utt_id} ({ark_offset}) 失败: {e}")
                        
        print(f"\n🎉 批量转换完成！共提取了 {count} 个音频文件到 '{output_dir}'。")
    except Exception as e:
        print(f"❌ 读取 SCP 文件出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据带有 offset 的 ark 路径或 scp 文件提取 wav")
    
    # 互斥参数组：要么提供单一字符串，要么提供 scp 文件
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--target", help="单条 ark 格式输入 (例如: /data/.../data_wav.ark:16920526)")
    group.add_argument("-s", "--scp", help="Kaldi 标准的 .scp 文件路径 (批量提取)")
    
    parser.add_argument("-o", "--output", default="./wav_output", help="输出文件夹或文件路径 (默认: ./wav_output)")
    parser.add_argument("-sr", "--samplerate", type=int, default=16000, help="默认采样率 (当ark中没有wav头时生效，默认: 16000)")
    
    args = parser.parse_args()

    if args.target:
        extract_single(args.target, args.output, args.samplerate)
    elif args.scp:
        extract_batch_scp(args.scp, args.output, args.samplerate)