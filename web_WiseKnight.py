# 首先导入 unsloth 以避免警告
import os
os.environ['UNSLOTH_IMPORT_FIRST'] = '1'

# 现在导入 unsloth
from unsloth import FastVisionModel

# 然后导入其他库
import argparse
import torch
import warnings
import gradio as gr
from queue import Queue
from threading import Thread
from PIL import Image
from transformers import TextStreamer
import base64
import re

warnings.filterwarnings('ignore')


def init_model(model_path):
    """
    加载模型：优先尝试本地，失败则从远程下载
    """
    # 检查是否为本地路径
    is_local_path = os.path.exists(model_path)
    
    # 设置加载配置
    load_kwargs = {
        "load_in_4bit": True,
        "max_seq_length": 8192,
        "trust_remote_code": True,
    }
    
    if is_local_path:
        # 优先尝试本地加载
        print(f"尝试加载本地模型: {model_path}")
        try:
            # 设置离线模式
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_path,
                local_files_only=True,  # 强制本地模式
                **load_kwargs
            )
            print("☑️  本地模型加载成功")
            
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试从远程下载模型...")
            
            # 清除离线设置，允许网络连接
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_HUB_OFFLINE', None)
            
            try:
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name=model_path,
                    local_files_only=False,  # 允许远程下载
                    **load_kwargs
                )
                print("☑️  远程模型下载并加载成功")
                
            except Exception as e2:
                raise RuntimeError(f"远程模型加载失败: {e2}")
    
    else:
        # 直接尝试从远程加载（可能是Hugging Face模型ID）
        print(f"尝试从远程加载模型: {model_path}")
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_path,
                local_files_only=False,
                **load_kwargs
            )
            print("☑️  远程模型加载成功")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    # 启用推理模式
    try:
        FastVisionModel.for_inference(model)
        print("☑️  推理模式已启用")
    except Exception as e:
        print(f"警告：启用推理模式失败，但模型已加载: {e}")
    
    print("☑️  模型加载完毕！")
    return model.eval().to(args.device), tokenizer


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def prepare_messages(image, prompt):
    """
    准备 Unsloth 格式的消息
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    return messages


def chat(prompt, current_image_path, show_think=False):
    """
    与模型聊天，分析图片
    """
    if not current_image_path:
        yield "错误：图片不能为空。"
        return
    
    try:
        image = Image.open(current_image_path).convert('RGB')
    except Exception as e:
        yield f"错误：无法加载图片 - {str(e)}"
        return
    
    # 准备消息
    messages = prepare_messages(image, prompt)
    
    # 使用和.ipynb中相同的方式处理输入
    try:
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # 处理图像输入
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(args.device)
    except Exception as e:
        print(f"处理输入失败: {e}")
        yield f"错误：处理输入失败 - {str(e)}"
        return
    
    queue = Queue()
    streamer = CustomStreamer(tokenizer, queue)
    
    def _generate():
        with torch.no_grad():
            try:
                # 使用确定的生成参数，和.ipynb中保持一致
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_seq_len,
                    do_sample=False,  # 不使用随机采样
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    num_beams=1,
                    streamer=streamer,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as e:
                queue.put(f"错误：生成失败 - {str(e)}")
                queue.put(None)
    
    Thread(target=_generate).start()
    
    response = ''
    # 无论是否显示思考过程，都使用流式输出以保持UI一致
    while True:
        text = queue.get()
        if text is None:
            break
        response += text
        yield response


def load_logo_base64(logo_path):
    """加载logo图片并转换为base64格式"""
    if not logo_path or not os.path.exists(logo_path):
        return None
    
    try:
        with open(logo_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # 检测图片格式
        if logo_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif logo_path.lower().endswith('.jpg') or logo_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif logo_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        else:
            mime_type = 'image/png'
        
        return f"data:{mime_type};base64,{img_base64}"
    except Exception as e:
        print(f"加载logo图片失败: {e}")
        return None


def format_chat_message(message, show_think=False):
    """格式化聊天消息，根据是否显示思考过程来提取标签并美化显示"""
    if not message:
        return ""
    
    message = message.strip()
    
    # 如果消息格式正确，提取think和label标签
    think_content = ""
    label_content = ""
    
    # 提取<think>标签内容
    if "<|think|>" in message:
        think_pattern = r'<\|think\|>(.*?)<\|think\|>'
        think_match = re.search(think_pattern, message, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
    
    # 提取<label>标签内容
    if "<|label|>" in message:
        label_pattern = r'<\|label\|>(.*?)<\|label\|>'
        label_match = re.search(label_pattern, message, re.DOTALL)
        if label_match:
            label_content = label_match.group(1).strip()
    
    formatted_message = ""

    # helper: hex (#rrggbb) -> rgba(r,g,b,a)
    def hex_to_rgba(hex_color, alpha=1.0):
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                return f'rgba(0,0,0,{alpha})'
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'
        except Exception:
            return f'rgba(0,0,0,{alpha})'
    
    # 格式化think内容（仅当show_think为True且有思考内容时显示）
    if think_content and show_think:
        lines = [line.strip() for line in think_content.split('\n') if line.strip()]
        formatted_think = ""
        for i, line in enumerate(lines, 1):
            # 保留原始编号格式
            if line.startswith(f"{i}.") or re.match(r'^\d+\.', line):
                formatted_think += f"{line}<br>"
            else:
                formatted_think += f"{i}. {line}<br>"
        
        formatted_message += f"""
        <div style=\"background: rgba(147, 51, 234, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #9333ea; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);\">\n            <div style=\"display: flex; align-items: center; margin-bottom: 15px;\">\n                <div style=\"background: linear-gradient(135deg, #9333ea 0%, #7c3aed 100%); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 4px rgba(147, 51, 234, 0.3);\">\n                    <span style=\"color: white; font-size: 18px; font-weight: bold;\">🧠</span>\n                </div>\n                <span style=\"font-weight: bold; color: #9333ea; font-size: 18px; text-shadow: 0 1px 2px rgba(0,0,0,0.1);\">已思考</span>\n            </div>\n            <div style=\"color: #374151; line-height: 1.7; font-size: 15px; padding-left: 10px;\">\n                {formatted_think}\n            </div>\n        </div>\n        """
    
    # 格式化label内容（总是显示）
    if label_content:
        # 根据标签设置不同的颜色
        color_map = {
            "非楼道": "#6b7280",    # 灰色
            "高风险": "#ef4444",    # 红色
            "中风险": "#f59e0b",    # 橙色
            "低风险": "#10b981",    # 绿色
            "无风险": "#3b82f6",    # 蓝色
        }
        label_color = color_map.get(label_content, "#9333ea")
        
        # 获取风险等级含义描述
        risk_desc = {
            "非楼道": "非楼道场景",
            "高风险": "存在严重安全隐患",
            "中风险": "存在一定安全隐患",
            "低风险": "基本安全",
            "无风险": "非常安全"
        }.get(label_content, "未知风险等级")
        
        # 计算半透明背景和阴影颜色
        rgba_bg = hex_to_rgba(label_color, 0.08)
        rgba_shadow = hex_to_rgba(label_color, 0.3)
        
        formatted_message += f"""
        <div style=\"background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%); padding: 25px; border-radius: 15px; border: 3px solid {label_color}; text-align: center; box-shadow: 0 6px 20px {rgba_shadow};\">\n            <div style=\"display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;\">\n                <div style=\"background: {label_color}; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 15px; box-shadow: 0 4px 8px {rgba_shadow};\">\n                    <span style=\"color: white; font-size: 24px;\">🏷️</span>\n                </div>\n                <span style=\"font-size: 22px; font-weight: bold; color: #1f2937; margin-bottom: 8px;\">检测结果</span>\n                <span style=\"font-size: 16px; color: #6b7280;\">{risk_desc}</span>\n            </div>\n            <div style=\"font-size: 32px; font-weight: bold; color: {label_color}; padding: 20px; background: {rgba_bg}; border-radius: 12px; display: inline-block; min-width: 180px; margin-bottom: 15px; border: 2px solid {label_color};\">\n                {label_content}\n            </div>\n            <div style=\"margin-top: 15px; font-size: 14px; color: #6b7280; font-style: italic;\">\n                消防隐患等级\n            </div>\n        </div>\n        """
    
    # 如果没有格式化内容，显示原始消息（这通常发生在模型直接输出标签的情况下）
    if not formatted_message:
        # 尝试从原始消息中提取标签
        direct_label_match = re.search(r'(非楼道|高风险|中风险|低风险|无风险)', message)
        if direct_label_match:
            label_content = direct_label_match.group(1)
            label_color = {
                "非楼道": "#6b7280",
                "高风险": "#ef4444",
                "中风险": "#f59e0b",
                "低风险": "#10b981",
                "无风险": "#3b82f6"
            }.get(label_content, "#9333ea")
            
            risk_desc = {
                "非楼道": "非楼道场景",
                "高风险": "存在严重安全隐患",
                "中风险": "存在一定安全隐患",
                "低风险": "基本安全",
                "无风险": "非常安全"
            }.get(label_content, "未知风险等级")
            
            rgba_bg = hex_to_rgba(label_color, 0.08)
            rgba_shadow = hex_to_rgba(label_color, 0.3)
            
            formatted_message += f"""
            <div style=\"background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%); padding: 25px; border-radius: 15px; border: 3px solid {label_color}; text-align: center; box-shadow: 0 6px 20px {rgba_shadow};\">\n                <div style=\"display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;\">\n                    <div style=\"background: {label_color}; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 15px; box-shadow: 0 4px 8px {rgba_shadow};\">\n                        <span style=\"color: white; font-size: 24px;\">🏷️</span>\n                    </div>\n                    <span style=\"font-size: 22px; font-weight: bold; color: #1f2937; margin-bottom: 8px;\">检测结果</span>\n                    <span style=\"font-size: 16px; color: #6b7280;\">{risk_desc}</span>\n                </div>\n                <div style=\"font-size: 32px; font-weight: bold; color: {label_color}; padding: 20px; background: {rgba_bg}; border-radius: 12px; display: inline-block; min-width: 180px; margin-bottom: 15px; border: 2px solid {label_color};\">\n                    {label_content}\n                </div>\n                <div style=\"margin-top: 15px; font-size: 14px; color: #6b7280; font-style: italic;\">\n                    消防隐患等级\n                </div>\n            </div>\n            """
        else:
            # 如果没有识别到标签，显示原始消息
            formatted_message = f"""
            <div style=\"padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); color: #374151; border-radius: 12px; border: 1px solid #cbd5e1; box-shadow: 0 2px 8px rgba(0,0,0,0.05);\">\n                <div style=\"display: flex; align-items: center; margin-bottom: 10px;\">\n                    <span style=\"font-weight: bold; color: #64748b;\">📋 分析结果：</span>\n                </div>\n                <div style=\"font-size: 15px; line-height: 1.6;\">\n                    {message}\n                </div>\n            </div>\n            """
    
    return formatted_message


def format_think_message_during_analysis(message, show_think):
    """在分析过程中格式化思考消息（根据是否显示思考过程）"""
    if not message or not show_think:
        return ""
    
    message = message.strip()
    
    # 如果消息以<|think|>开头，我们认为是思考内容
    if message.startswith("<|think|>") or "<|think|>" in message:
        # 先找到第一个<|think|>之后的内容
        think_start = message.find("<|think|>") + 9  # 9是"<|think|>"的长度
        
        # 然后找第一个<|think|>结束标签（从think_start开始找）
        think_end = message.find("<|think|>", think_start)
        
        think_content = ""
        if think_end != -1:
            # 找到了结束标签
            think_content = message[think_start:think_end].strip()
        else:
            # 没有结束标签，取从开始到字符串末尾
            # 但要排除可能已经出现的<|label|>
            label_start = message.find("<|label|>", think_start)
            if label_start != -1:
                think_content = message[think_start:label_start].strip()
            else:
                think_content = message[think_start:].strip()
        
        # 格式化思考内容
        if think_content:
            lines = [line.strip() for line in think_content.split('\n') if line.strip()]
            formatted_think = ""
            
            for i, line in enumerate(lines, 1):
                # 检查是否已经有编号
                if re.match(r'^\d+\.', line):
                    formatted_think += f"<div style='margin-bottom: 8px; padding: 8px 12px; background: rgba(147, 51, 234, 0.05); border-radius: 8px; border-left: 3px solid #9333ea;'>{line}</div>"
                else:
                    formatted_think += f"<div style='margin-bottom: 8px; padding: 8px 12px; background: rgba(147, 51, 234, 0.05); border-radius: 8px; border-left: 3px solid #9333ea;'>{i}. {line}</div>"
            
            # 只有当有格式化内容时才返回
            if formatted_think:
                return f"""
                <div style=\"background: rgba(147, 51, 234, 0.08); padding: 20px; border-radius: 12px; border-left: 5px solid #9333ea; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);\">
                    <div style=\"display: flex; align-items: center; margin-bottom: 15px;\">
                        <div style=\"background: linear-gradient(135deg, #9333ea 0%, #7c3aed 100%); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 4px rgba(147, 51, 234, 0.3);\">
                            <span style=\"color: white; font-size: 18px; font-weight: bold;\">🧠</span>
                        </div>
                        <span style=\"font-weight: bold; color: #9333ea; font-size: 18px; text-shadow: 0 1px 2px rgba(0,0,0,0.1);\">正在思考</span>
                    </div>
                    <div style=\"color: #374151; line-height: 1.7; font-size: 15px;\">
                        {formatted_think}
                    </div>
                </div>
                """
    
    return ""


def get_prompt_without_think():
    """获取不显示思考过程的prompt版本（加上空的think标签）"""
    base_prompt = """你将获得该一张图像并逐步分析该图像的标签，标签可为 "非楼道"、"高风险"、"中风险"、"低风险" 或 "无风险"。
    评估规则：
    如果图像不属于楼道场景，则正确标签为 "非楼道"。
    如果图像属于楼道场景，则评估火灾风险：
    a. 如果发现任何停放的电动自行车、电池充电设备或临时充电（例如"飞线充电"），则标记为 "高风险"。
    b. 否则，如果有大量杂物，且满足以下任一条件：（i）严重阻碍通行，或（ii）包括明显可燃物品（如纸箱、木制家具、布艺家具、泡沫箱等），则标记为 "中风险"。
    c. 否则，如果有少量物品或摆放整齐，仅对通行造成轻微影响，则标记为 "低风险"。
    d. 否则，走廊干净无存放物，则标记为 "无风险"。
    任务：
    给定该图像，直接输出最终标签结果，不要输出思考链。
    输出格式：
    <|think|><|think|>
    <|label|>最终输出标签结果<|label|>
    不要额外文本，仅输出最终结果。"""
    
    return base_prompt


def get_prompt_with_think():
    """获取显示思考过程的prompt版本（原始版本）"""
    base_prompt = """你将获得该一张图像并逐步分析该图像的标签，标签可为 "非楼道"、"高风险"、"中风险"、"低风险" 或 "无风险"。
    评估规则：
    如果图像不属于楼道场景，则正确标签为 "非楼道"。
    如果图像属于楼道场景，则评估火灾风险：
    a. 如果发现任何停放的电动自行车、电池充电设备或临时充电（例如"飞线充电"），则标记为 "高风险"。
    b. 否则，如果有大量杂物，且满足以下任一条件：（i）严重阻碍通行，或（ii）包括明显可燃物品（如纸箱、木制家具、布艺家具、泡沫箱等），则标记为 "中风险"。
    c. 否则，如果有少量物品或摆放整齐，仅对通行造成轻微影响，则标记为 "低风险"。
    d. 否则，走廊干净无存放物，则标记为 "无风险"。
    任务：
    给定该图像，输出一个编号列表，展示你的思考链，明确指出你观察到的视觉特征，最终得该图像的标签。你需要确思考链结果被包裹在<|think|>标签中，并在确保最终输出标签结果被包裹在<|label|>标签中。
    输出格式：
    <|think|>1.推理步骤一……
    2.推理步骤二……
    …
    n. 最终论证步骤……<|think|>
    <|label|>最终输出标签结果<|label|>
    不要额外文本，仅输出推理步骤和最终结果。"""
    
    return base_prompt


def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    """
    启动Gradio服务器
    """
    # 加载logo图片
    logo_path = args.logo_path
    logo_data = None
    
    if logo_path and os.path.exists(logo_path):
        logo_data = load_logo_base64(logo_path)
    
    if not logo_data:
        print("未找到指定logo图片，使用默认样式。")
    
    with gr.Blocks(theme=gr.themes.Soft(), title="消防隐患识别智慧骑士系统") as demo:      
        # 标题区域
        if logo_data:
            gr.HTML(f"""
                <div style=\"text-align: center; margin-bottom: 2.5rem; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2); position: relative; overflow: hidden;\">\n                    <div style=\"position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(255,255,255,0.1); border-radius: 50%;\"></div>\n                    <div style=\"position: absolute; bottom: -80px; left: -80px; width: 250px; height: 250px; background: rgba(255,255,255,0.05); border-radius: 50%;\"></div>\n                    \n                    <div style=\"display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; position: relative; z-index: 2;\">\n                        <img src=\"{logo_data}\" style=\"height: 100px; width: 100px; object-fit: contain; margin-right: 2rem; border-radius: 50%; border: 5px solid rgba(255,255,255,0.8); box-shadow: 0 6px 15px rgba(0,0,0,0.3);\">\n                        <div style=\"text-align: left;\">\n                            <h1 style=\"font-size: 44px; color: white; margin: 0; font-weight: 900; text-shadow: 2px 3px 6px rgba(0,0,0,0.4); letter-spacing: 0.5px;\">消防隐患识别智慧骑士系统</h1>\n                            <p style=\"font-size: 20px; color: rgba(255,255,255,0.95); margin: 10px 0 0 0; font-style: italic; font-weight: 300;\">视觉大模型分析 · 识别楼道消防隐患</p>\n                        </div>\n                    </div>\n                    <div style=\"position: relative; z-index: 2; margin-top: 1rem;\">\n                        <div style=\"display: inline-block; background: rgba(255,255,255,0.15); padding: 8px 20px; border-radius: 25px; border: 1px solid rgba(255,255,255,0.3);\">\n                            <span style=\"font-size: 14px; color: rgba(255,255,255,0.9); font-family: 'Courier New', monospace;\">\n                                💡 智能识别 · ⚠️ 风险分析 · 🛡️ 安全评估\n                            </span>\n                        </div>\n                    </div>\n                </div>\n            """)
        else:
            gr.HTML(f"""
                <div style=\"text-align: center; margin-bottom: 2.5rem; padding: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);\">\n                    <h1 style=\"font-size: 48px; color: white; margin: 0 0 15px 0; font-weight: 900; text-shadow: 2px 3px 8px rgba(0,0,0,0.4);\">\n                        🔥 消防隐患识别智慧骑士系统 🔥\n                    </h1>\n                    <p style=\"font-size: 22px; color: rgba(255,255,255,0.95); margin: 0 0 20px 0; font-style: italic; font-weight: 300;\">\n                        视觉大模型分析 · 识别楼道消防隐患\n                    </p>\n                    <div style=\"display: inline-block; background: rgba(255,255,255,0.2); padding: 10px 25px; border-radius: 30px; border: 2px solid rgba(255,255,255,0.4);\">\n                        <span style=\"font-size: 16px; color: white; font-weight: 500;\">\n                            💡 智能识别 · ⚠️ 风险分析 · 🛡️ 安全评估\n                        </span>\n                    </div>\n                </div>\n            """)
        
        # 使用Tab布局来均衡左右分布
        with gr.Row():
            # 左侧：图片上传和控制区域 (45%)
            with gr.Column(scale=5, min_width=400):
                with gr.Group():
                    gr.Markdown("### 📸 图片上传区域")
                    image_input = gr.Image(
                        type="filepath", 
                        label="",  # 清空标签文本
                        height=320,
                        interactive=True,
                        elem_id="image_upload",
                        sources=["upload"],  # 只保留上传功能
                        show_label=False  # 添加这行来彻底隐藏标签
                    )
                    gr.Markdown("""
                    <div style="text-align: center; margin-top: 10px; color: #6b7280; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;">
                        <span>支持 JPG、PNG、JPEG 格式，最大 10MB</span>
                    </div>
                    """)
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 分析设置")
                    # 添加"思考分析"复选框
                    think_analysis_checkbox = gr.Checkbox(
                        label="🧠 深度思考",
                        value=False,
                        info="勾选此选项将在分析过程中启用深度思考并实时显示思考步骤"
                    )
                
                with gr.Group():
                    gr.Markdown("### 🛠️ 系统操作")
                    with gr.Row():
                        submit_btn = gr.Button(
                            "🔍 开始分析", 
                            variant="primary", 
                            size="lg",
                            scale=2,
                            elem_id="analyze_btn"
                        )
                        clear_btn = gr.Button(
                            "🔄 重置系统", 
                            variant="secondary", 
                            size="lg",
                            scale=1,
                            elem_id="clear_btn"
                        )

                with gr.Accordion("📋 分析标准说明(点击即可查看)", open=False):
                    gr.Markdown("""
                    ### 风险等级定义：
                    
                    **🔴 高风险**：发现电动自行车、电池充电设备或飞线充电
                    
                    **🟠 中风险**：有大量杂物，且：
                    - 严重阻碍通行
                    - 或包含明显可燃物品（纸箱、木制家具、泡沫箱等）
                    
                    **🟢 低风险**：有少量物品或摆放整齐，仅轻微影响通行
                    
                    **🔵 无风险**：走廊干净无存放物
                    
                    **⚪ 非楼道**：图像不属于楼道场景
                    """)
            
            # 分析结果和系统状态 (55%)
            with gr.Column(scale=7, min_width=500):
                with gr.Group():
                    gr.Markdown("### 📝 视觉大模型分析")
                    status = gr.Textbox(
                        label="",
                        value="🆙 系统就绪，请上传楼道场景图片进行分析",
                        interactive=False,
                        elem_id="status_display",
                        lines=2
                    )
                    result_html = gr.HTML(
                        value="",
                        elem_id="analysis_result",
                    )

        # 底部信息栏
        gr.HTML("""
        <div style="margin-top: 30px; padding: 15px; background: #f8fafc; border-radius: 10px; border-top: 3px solid #667eea; text-align: center;">
            <div style="margin-top: 10px; font-size: 12px; color: #9ca3af;">
                © 消防隐患识别智慧骑士系统
            </div>
        </div>
        """)
        
        # 添加自定义CSS样式
        gr.HTML("""
        <style>      
        /* 确保所有中文和英文字符正常显示 */
        body, .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
        }
        </style>
        """)
        
        # 定义处理函数
        current_image_path = gr.State("")
        history_state = gr.State([])
        show_think_state = gr.State(False)
        
        def update_image_path(image):
            """更新当前图片路径"""
            if image:
                return image
            return ""
        
        def update_think_setting(show_think):
            """更新是否显示思考分析的设置"""
            return show_think
        
        def render_history_html(history):
            """
            将 [(user, bot_html)] 渲染为 HTML
            只显示用户消息；只有在 bot_msg 非 None 时才显示 model 的 HTML。
            """
            html = ""
            for user_msg, bot_msg in history:
                html += f"""
                <div style="margin-bottom: 18px;">
                    <div style="font-weight: 700; color: #374151; margin-bottom: 6px;">
                        {user_msg}
                    </div>
                """
                # 仅在有模型返回结果（非 None）时才显示
                if bot_msg is not None:
                    html += f"<div>{bot_msg}</div>"
                # 如果 bot_msg 是 None -> 不显示任何占位文本（不闪现小字）
                html += "</div>"
            return html
        
        def analyze_image(image, show_think, history):
            """分析图片（简化版本，避免复杂逻辑）"""
            if not image:
                history.append(("系统", "❌ 错误：请先上传图片"))
                status_msg = "❌ 未上传图片，请先选择图片"
                return history, status_msg, history, show_think

            filename = os.path.basename(image) if image else "未知图片"
            user_message = f"📦 分析请求 —— 图片: {filename}"
            # 保持与原来一致：先插入一个占位（bot_msg 为 None）
            history.append((user_message, None))

            # 根据是否显示思考过程设置不同的状态消息和 prompt
            if show_think:
                status_msg = "⏳ 正在启用深度思考分析图片......"
                prompt = get_prompt_with_think()
            else:
                status_msg = "⏳ 正在分析图片......"
                prompt = get_prompt_without_think()

            # 初始界面
            html = render_history_html(history)
            yield html, status_msg, history, show_think

            # 调用聊天函数 - 传入相应的 prompt
            response_generator = chat(prompt, image, show_think)

            try:
                full_response = ""
                last_think_html = ""

                for response in response_generator:
                    full_response = response
                    
                    # 如果显示思考过程，尝试格式化思考内容
                    if show_think:
                        think_html = format_think_message_during_analysis(full_response, show_think)
                        # 只有当有新的思考内容时才更新显示
                        if think_html and think_html != last_think_html:
                            last_think_html = think_html
                            current_history = history[:-1] + [(user_message, think_html)]
                            html = render_history_html(current_history)
                            yield html, status_msg, history, show_think
                    else:
                        # 不显示思考过程，保持占位不变
                        pass

                # 分析完成，显示最终格式化结果
                formatted_response = format_chat_message(full_response, show_think)
                history = history[:-1] + [(user_message, formatted_response)]
                final_html = render_history_html(history)

                if show_think:
                    final_status = "✅ 分析完成！已显示完整的思考过程和最终评估结果"
                else:
                    final_status = "✅ 分析完成！已显示最终评估结果"
                yield final_html, final_status, history, show_think

            except Exception as e:
                error_msg = f"❌ 分析失败: {str(e)}"
                history = history[:-1] + [(user_message, error_msg)]
                final_html = render_history_html(history)
                yield final_html, f"❌ 分析过程出错", history, show_think

            return
        
        def clear_all():
            return None, False, "", "", [], False
        
        # 绑定事件
        image_input.change(
            fn=update_image_path,
            inputs=image_input,
            outputs=current_image_path
        ).then(
            fn=lambda img: "🆗 图片已上传，点击【开始分析】按钮进行图片分析" if img else "📤 等待上传图片",
            inputs=image_input,
            outputs=status
        )
        
        # 更新思考分析设置
        think_analysis_checkbox.change(
            fn=update_think_setting,
            inputs=think_analysis_checkbox,
            outputs=show_think_state
        )
        
        # 开始分析按钮
        submit_btn.click(
            fn=analyze_image,
            inputs=[image_input, show_think_state, history_state],
            outputs=[result_html, status, history_state, show_think_state]
        )
        
        # 清空按钮
        clear_btn.click(
            fn=clear_all,
            outputs=[
                image_input,
                think_analysis_checkbox,
                status,
                current_image_path,
                history_state,
                show_think_state
            ]
        ).then(
            fn=lambda: "🔄 系统已重置，请重新上传图片",
            outputs=status
        ).then(
            fn=lambda: "",
            outputs=result_html
        )
    
    print(f"🚀 消防隐患识别智慧骑士系统已启动")
    print(f"📁 使用的模型路径: {args.model_path}")
    print(f"🌐 访问地址: http://{server_name}:{server_port}")
    
    demo.launch(
        server_name=server_name, 
        server_port=server_port,
        share=args.share,
        quiet=True,
        show_error=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="消防隐患识别智慧骑士系统")
    
    # 参数
    parser.add_argument('--model_path', default="model/llama-3-2-11b-vision-instruct-4bit-r16-think/last_v2", type=str, 
                    help="模型路径")
    
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                       type=str, help="运行设备")
    parser.add_argument('--max_seq_len', default=4096, type=int, 
                       help="最大序列长度")
    parser.add_argument('--port', default=8888, type=int, 
                       help="服务器端口")
    parser.add_argument('--share', default=False, action='store_true',
                       help="是否创建公开可访问的链接")
    parser.add_argument('--logo_path', default="assets/images/logo.png", type=str,
                       help="logo图片路径")
    
    args = parser.parse_args()

    
    # 初始化模型
    model, tokenizer = init_model(args.model_path)
    
    # 启动服务器
    launch_gradio_server(server_name="0.0.0.0", server_port=args.port)