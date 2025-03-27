from openai import OpenAI
import os
import sys

client = OpenAI(api_key=api_key)

def get_gpt_response(prompt, model="gpt-3.5-turbo"):
    """发送提示到GPT并获取回复"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"错误: {str(e)}"

def main():
    """主函数，处理命令行参数或交互式输入"""
    # 如果有命令行参数，将其作为提示
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response = get_gpt_response(prompt)
        print(f"\n{response}\n")
    else:
        # 否则进入交互模式
        print("输入 'exit' 或 'quit' 结束程序")
        while True:
            try:
                prompt = input("\n请输入您的提示 > ")
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                if prompt.strip():
                    print("\n正在思考...\n")
                    response = get_gpt_response(prompt)
                    print(f"{response}\n")
            except KeyboardInterrupt:
                print("\n程序已被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
