import gradio as gr
from openai import OpenAI
from urllib.parse import urlparse
import base64
import os

# 配置 SGLang 客户端
client = OpenAI(
    base_url="http://127.0.0.1:6008/v1",
    api_key="EMPTY"
)

def get_model_name():
    try:
        models = client.models.list()
        return models.data[0].id if models.data else "qwen3.8-27b-fp8"
    except Exception:
        return "qwen3.8-27b-fp8"

MODEL_NAME = get_model_name()

def chat(messages: list, disable_thinking: bool, thinking_budget: int):
    """
    流式调用 SGLang，messages 已经是完整的对话列表
    """
    # 构建 extra_body
    extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": not disable_thinking
        }
    }
    
    if thinking_budget > 0 and not disable_thinking:
        extra_body["chat_template_kwargs"]["thinking_budget"] = int(thinking_budget)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # 直接使用传入的列表
        stream=True,
        temperature=0.7,
        max_tokens=8192,
        extra_body=extra_body
    )
    
    partial_message = ""
    thinking_started = False
    
    for chunk in stream:
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        
        reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None) or ""
        content = delta.content or ""
        
        if reasoning:
            if not thinking_started:
                partial_message += "<details open>\n<summary> 推理过程 (点击收起)</summary>\n\n"
                thinking_started = True
            partial_message += reasoning
        
        if content:
            if thinking_started:
                partial_message += "\n\n</details>\n\n"
                thinking_started = False
            partial_message += content
            
        if chunk.choices[0].finish_reason is not None and thinking_started:
            partial_message += "\n\n</details>\n\n"
            thinking_started = False
            
        yield partial_message

def convert_gradio_to_openai(gradio_history):
    openai_messages = []
    for msg in gradio_history:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
            
        if isinstance(content, list):
            new_content = []
            for item in content:
                if "text" in item:
                    new_content.append({"type": "text", "text": item["text"]})
                    
                elif "image" in item:
                    abs_path = os.path.abspath(item["image"])
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{abs_path}"}
                    })
                    
                elif "file" in item:
                    abs_path = os.path.abspath(item["file"])
                    new_content.append({
                        "type": "video_url", 
                        "video_url": {"url": f"file://{abs_path}"}
                    })
                    
            openai_messages.append({"role": role, "content": new_content})
            
    return openai_messages

def encode_image_to_base64(image_path):
    """将图片转换为 base64 编码"""
    if image_path is None:
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_video_to_base64(video_path):
    """将视频转换为 base64 编码"""
    if video_path is None:
        return None
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

with gr.Blocks(title="Qwen3.8-27B-FP8 Chat") as demo:
    # 先放一个空的 Markdown，等加载后填充
    url_display = gr.Markdown("## 💬 Qwen3.8-27B (FP8 Thinking)\n\n### 需要先开放端口才能使用（网页使用6006端口，API使用6008端口）！开放端口参考： https://gitcode.com/aistudent/ai-mirror-list/blob/main/README.md#autodl%E5%BC%80%E6%94%BE%E7%AB%AF%E5%8F%A3%E6%95%99%E7%A8%8B  \n\n> 启动状态检查（使用6008端口访问，启动成功会显示: SGLang is running）)\n\n \n\nAgent接入指南在**页面底部**")
    chatbot = gr.Chatbot(height=500, latex_delimiters=[])
    
    msg = gr.Textbox(
        placeholder="输入消息... 支持提问代码、数学、逻辑推理等问题",
        lines=3,
        autofocus=True,
    )
    file_paths_state = gr.State([])
    
    with gr.Accordion("📎 上传文件或媒体（可选）", open=False):
        with gr.Row():
            image_input = gr.Image(
                label="上传图片",
                type="filepath",
                sources=["upload", "clipboard"],
                height=200
            )
            video_input = gr.File(
                label="上传视频",
                file_types=["video"],
                file_count="single"
            )
    
    # 【新增】：思考模式与推理强度控制面板
    with gr.Row():
        disable_thinking = gr.Checkbox(label="⚡ 禁用思考 (快速模式)", value=False)
        thinking_budget = gr.Slider(
            minimum=0, 
            maximum=8192, 
            step=256, 
            value=0, 
            label="🧠 思考预算 (0为无限制)",
            info="限制推理过程的最大Token数，数值越小回答越快"
        )
    
    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话")

    # 使用说明 Markdown
    gr.Markdown(f"""## 🚀 OpenClaw/Codex/Claude Code/Agent 使用指南



    1. OpenAI格式
    2. v1服务地址: （开放端口参考: https://gitcode.com/aistudent/ai-mirror-list/blob/main/README.md#autodl%E5%BC%80%E6%94%BE%E7%AB%AF%E5%8F%A3%E6%95%99%E7%A8%8B ）（端口 6008 地址后面追加 /v1 ，示例: https://xxxx.westd.seetacloud.com:8443/v1 ）
    3. 模型ID（名称）： qwen3.8-27b-fp8
    4. 开始使用。

### 注意：模型本身支持图片、视频理解，但部分Agent工具可能会将模型识别为文本模型，因此可能需要手动设置模型属性（Qwen Code可以参考下面的设置进行手动配置）。

### Qwen Code设置参考

~/.qwen/settings.json

```
 {{
    "id": "qwen3.8-27b-fp8",
    "name": "qwen3.8-27b-fp8",
    "baseUrl": "https://xxxxx你自己的地址.westd.seetacloud.com/v1/",
    "envKey": "QWEN_CUSTOM_API_KEY_OPENAI_97F",
    "generationConfig": {{
      "timeout": 300000,
      "contextWindowSize": 262144,
      "modalities": {{
        "image": true,
        "video": true
      }},
      "extra_body": {{
        "chat_template_kwargs": {{
          "enable_thinking": true
        }}
      }}
    }}
  }},
```

## 查看SGlang启动日志

进入JupyterLab，然后在终端中输入(/root目录):

> tail -f /root/qwen38_start.log

即可查看SGlang启动日志.

启动成功会显示如下日志:
```
[2026-08-23 12:08:42] INFO:     Started server process [13207]
[2026-08-23 12:08:42] INFO:     Waiting for application startup.
[2026-08-23 12:08:42] Using default chat sampling params from model generation config: {{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}}
[2026-08-23 12:08:42] INFO:     Application startup complete.
[2026-08-23 12:08:42] INFO:     Uvicorn running on http://0.0.0.0:6008 (Press CTRL+C to quit)
```
    """)

    def user_submit(message, history, image_file, video_file, file_paths):
        if not message.strip() and image_file is None and video_file is None:
            return "", history, None, None, file_paths
        
        # Chatbot 中只显示纯文本 + 文件标记
        display_content = message.strip() or ""
        if image_file is not None:
            display_content += "\n📷 [已附加图片]"
        if video_file is not None:
            display_content += "\n🎬 [已附加视频]"
        
        # 保存文件路径到 State
        file_paths.append({"image": image_file, "video": video_file})
        
        # Chatbot 只接收纯文本字符串
        new_history = history + [{"role": "user", "content": display_content}]
        return "", new_history, None, None, file_paths

    def bot_respond(history, disable_think, think_budget, file_paths):
        if not history:
            yield history
            return
        
        # 构建 API 消息（包含多模态内容）
        api_messages = []
        user_msg_index = 0
        
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            
            # 【关键修复】：兼容 Gradio 6.x 的 messages 格式，安全提取纯文本
            if isinstance(content, list):
                # Gradio 6.x 会将文本包装成 [{"type": "text", "text": "..."}]
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                # 如果没找到 text 类型，就强行转为字符串（兜底）
                actual_text = text_content if text_content else str(content)
            else:
                # 如果是纯字符串，直接使用
                actual_text = str(content) if content else ""

            if role == "user" and user_msg_index < len(file_paths):
                file_info = file_paths[user_msg_index]
                # 使用提取出的干净字符串 actual_text
                content_parts = [{"type": "text", "text": actual_text}]
                
                # 添加图片
                if file_info.get("image"):
                    abs_path = os.path.abspath(file_info["image"])
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{abs_path}"}
                    })
                
                # 添加视频
                if file_info.get("video"):
                    abs_path = os.path.abspath(file_info["video"])
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": f"file://{abs_path}"}
                    })
                
                api_messages.append({"role": "user", "content": content_parts})
                user_msg_index += 1
            else:
                # 对于 assistant 或 system，直接传递提取出的纯文本
                api_messages.append({"role": role, "content": actual_text})
        
        # 添加 assistant 占位
        assistant_entry = {"role": "assistant", "content": ""}
        history.append(assistant_entry)
        
        # 发送给 API（不包含最后的空 assistant）
        for partial in chat(api_messages, disable_think, think_budget):
            history[-1]["content"] = partial
            yield history

    # 绑定事件，确保 UI 组件的值能正确传入 bot_respond
    msg.submit(
        user_submit, 
        [msg, chatbot, image_input, video_input, file_paths_state], 
        [msg, chatbot, image_input, video_input, file_paths_state]
    ).then(
        bot_respond, 
        [chatbot, disable_thinking, thinking_budget, file_paths_state], 
        chatbot
    )
    
    submit_btn.click(
        user_submit, 
        [msg, chatbot, image_input, video_input, file_paths_state], 
        [msg, chatbot, image_input, video_input, file_paths_state]
    ).then(
        bot_respond, 
        [chatbot, disable_thinking, thinking_budget, file_paths_state], 
        chatbot
    )
    
    clear_btn.click(lambda: ([], []), outputs=[chatbot, file_paths_state])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        theme=gr.themes.Soft(),
        css="""
        /* 思考块样式美化 */
        .gradio-container details {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #0d6efd;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 16px 0;
            font-size: 14px;
            line-height: 1.6;
            color: #495057;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .gradio-container details:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .gradio-container details summary {
            font-weight: 600;
            color: #0d6efd;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            list-style: none;
            outline: none;
        }
        
        .gradio-container details summary::-webkit-details-marker {
            display: none;
        }
        
        .gradio-container details summary::before {
            content: "▶";
            font-size: 12px;
            transition: transform 0.2s;
        }
        
        .gradio-container details[open] summary::before {
            transform: rotate(90deg);
        }
        
        .gradio-container details > :not(summary) {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed #ced4da;
        }
        
        /* 代码块样式 */
        .gradio-container pre { 
            border-radius: 8px !important; 
            margin: 12px 0 !important;
        }
        .gradio-container code { 
            font-size: 14px !important; 
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        }
        """
    )