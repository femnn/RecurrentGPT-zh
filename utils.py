import re
import openai

def get_api_response(content: str, max_tokens=None):

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and creative assistant for writing novel.\
                You are must always in Chinese.重要，你需要使用中文与我进行交流。'
        }, {
            'role': 'user',
            'content': content,
        }],
        temperature=0.4,  
        max_tokens=max_tokens
    )
    
    return response['choices'][0]['message']['content']

def get_content_between_a_b(a, b, text):
    match = re.search(f"{a}(.*?)\n{b}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # 处理找不到匹配内容的情况
        return "翻译时出现错误请重试"  # 或者返回其他默认值或采取其他的处理方式



def get_init(init_text=None,text=None,response_file=None):
    """
    init_text: if the title, outline, and the first 3 paragraphs are given in a .txt file, directly read
    text: if no .txt file is given, use init prompt to generate
    """
    if not init_text:
        response = get_api_response(text)
        print(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Init output here:\n{response}\n\n")
    else:
        with open(init_text,'r',encoding='utf-8') as f:
            response = f.read()
        f.close()
    paragraphs = {
        "name":"",
        "Outline":"",
        "Paragraph 1":"",
        "Paragraph 2":"",
        "Paragraph 3":"",
        "Summary": "",
        "Instruction 1":"",
        "Instruction 2":"", 
        "Instruction 3":""    
    }
    paragraphs['name'] = get_content_between_a_b('名称：','概述：',response)
    
    paragraphs['Paragraph 1'] = get_content_between_a_b('段落 1：','段落 2：',response)
    paragraphs['Paragraph 2'] = get_content_between_a_b('段落 2：','段落 3：',response)
    paragraphs['Paragraph 3'] = get_content_between_a_b('段落 3：','总结',response)
    paragraphs['Summary'] = get_content_between_a_b('总结：','指令 1',response)
    paragraphs['Instruction 1'] = get_content_between_a_b('指令 1：','指令 2：',response)
    paragraphs['Instruction 2'] = get_content_between_a_b('指令 2：','指令 3：',response)
    lines = response.splitlines()
    # content of Instruction 3 may be in the same line with I3 or in the next line
    if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
        paragraphs['Instruction 3'] = lines[-1][len("Instruction 3:"):]
    elif lines[-1] != '\n':
        paragraphs['Instruction 3'] = lines[-1]
    # Sometimes it gives Chapter outline, sometimes it doesn't
    for line in lines:
        if line.startswith('Chapter'):
            paragraphs['Outline'] = get_content_between_a_b('概述：','Chapter',response)
            break
    if paragraphs['Outline'] == '':
        paragraphs['Outline'] = get_content_between_a_b('概述：','段落',response)


    return paragraphs

def get_chatgpt_response(model,prompt):
    response = ""
    for data in model.ask(prompt):
        response = data["message"]
    model.delete_conversation(model.conversation_id)
    model.reset_chat()
    return response


def parse_instructions(instructions):
    output = ""
    for i in range(len(instructions)):
        output += f"{i+1}. {instructions[i]}\n"
    return output
