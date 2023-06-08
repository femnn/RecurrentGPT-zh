import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
import re

# from urllib.parse import quote_plus
# from pymongo import MongoClient

# uri = "mongodb://%s:%s@%s" % (quote_plus("xxx"),
#                               quote_plus("xxx"), "localhost")
# client = MongoClient(uri, maxPoolSize=None)
# db = client.recurrentGPT_db
# log = db.log

_CACHE = {}


# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

def init_prompt(novel_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
Please write a {novel_type} novel{description} with 50 chapters. Follow the format below precisely:

Begin with the name of the novel.
Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.
Write the first three paragraphs with their indication of the novel based on your outline. Write in a novelistic style and take your time to set the scene.
Write a summary that captures the key information of the three paragraphs.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
名称： <name of the novel>
概述: <outline for the first chapter>
段落 1： <content for paragraph 1>
段落 2： <content for paragraph 2>
段落 3： <content for paragraph 3>
总结： <content of summary>
指令 1： <content for instruction 1>
指令 2： <content for instruction 2>
指令 3：<content for instruction 3>

Make sure to be precise and follow the output format strictly.
非常重要！请将输出信息内容全部转化为中文，注意要符合中文母语的语法和用词习惯。


"""

def init(novel_type, description, request: gr.Request):
    if novel_type == "":
        novel_type = "科幻故事"
    global _CACHE
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    # prepare first init
    init_paragraphs = get_init(text=init_prompt(novel_type,description))
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""标题: {init_paragraphs['name']}

梗概: {init_paragraphs['Outline']}

段落:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(short_memory, long_memory, instruction1, instruction2, instruction3, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        #randomly select one instruction out of three
        instruction_index = random.randint(0,2)
        output['output_instruction'] = [instruction1, instruction2, instruction3][instruction_index]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(short_memory, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], parse_instructions(writer.long_memory), current_paras + '\n\n' + writer.output['input_paragraph'], *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("指令 ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # RecurrentGPT中文版
    汉化by:AI话聊室,  
    
    欢迎关注同名[微信公众号](http://m.qpic.cn/psc?/V11KbKJY37bMGc/ruAMsa53pVQWN7FLK88i5m2ntK4DhKP.Rf0Q2ejoAmnC*jFdjWtdNB4LJo9qcVsngIT9tADKFN5y89c*ZOZ90km.jG15KOhx*YGLjSZsQts!/b&bo=WAFYAQAAAAABByA!&rf=viewer_4)

    可以根据题目和简介自动续写文章
    
    也可以手动选择剧情走向进行续写 
    """)
    with gr.Tab("自动剧情"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button(
                    "点击开始运行", variant="primary")
                gr.Examples(["科幻故事", "青春伤痛文学", "爱到死去活来", "搞笑",
                            "幽默", "鬼故事", "喜剧", "童话", "魔法世界", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="文章内容", max_lines=21, lines=21)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### 剧情模型\n")
                    short_memory = gr.Textbox(
                        label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="长期记忆 (可编辑)", max_lines=6, lines=6)
                    # long_memory = gr.Dataframe(
                    #     # label="Long-Term Memory (editable)",
                    #     headers=["Long-Term Memory (editable)"],
                    #     datatype=["str"],
                    #     row_count=3,
                    #     max_rows=3,
                    #     col_count=(1, "fixed"),
                    #     type="array",
                    # )
                with gr.Box():
                    gr.Markdown("### 选项模型\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令1(可编辑)", max_lines=4, lines=4)
                        instruction2 = gr.Textbox(
                            label="指令2(可编辑)", max_lines=4, lines=4)
                        instruction3 = gr.Textbox(
                            label="指令3(可编辑)", max_lines=4, lines=4)
                    selected_plan = gr.Textbox(
                        label="选项说明 (来自上一步)", max_lines=2, lines=2)

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])

    with gr.Tab("自选择剧情"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button(
                    "点击开始运行", variant="primary")
                gr.Examples(["科幻小说", "爱情小说", "推理小说", "奇幻小说",
                            "玄幻小说", "恐怖", "悬疑", "惊悚", "武侠小说", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="文章内容 (可编辑)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### 剧情模型\n")
                    short_memory = gr.Textbox(
                        label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="长期记忆 (可编辑)", max_lines=6, lines=6)
                with gr.Box():
                    gr.Markdown("### 选项模型\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令1", max_lines=3, lines=3, interactive=False)
                        instruction2 = gr.Textbox(
                            label="指令2", max_lines=3, lines=3, interactive=False)
                        instruction3 = gr.Textbox(
                            label="指令3", max_lines=3, lines=3, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["指令 1", "指令 2", "指令 3"], label="指令 选择",)
                                                    #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="在上一步骤中被选择的 (可编辑)", max_lines=5, lines=5)

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True,
                server_name="0.0.0.0", show_api=False)