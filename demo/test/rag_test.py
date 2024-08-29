import json
import logging
import requests
from demo.pipeline.qa import read_jsonl, save_answers_test
import os


# queries = read_jsonl("../question-pdd.jsonl")


# llama_index
def llama_index():
    queries = read_jsonl("../sichuan-jike.jsonl")
    results, references_list = [], []
    url = "http://127.0.0.1:9111/chaxun"
    for query in queries:
        data = {
            "query": query["query"],
            "reranker_top": 7
        }
        response = requests.post(url, json=data)
        content = response.content.decode('utf-8')
        content = json.loads(content)
        output = content["res"]["text"]
        references = content["res"]["quote"]
        print(query["id"], "---->\n", output)
        results.append(output)
        references_list.append(references)
        # break
    print("开始保存llama_index结果。")

    # save_answers_test([queries[0]], results, references_list, path="./answers_llama_index.jsonl")
    save_answers_test(queries, results, references_list, path="./answers_llama_index-20240815-2.jsonl")
    print("llama_index save completed!")


# omega-gpt
def omega_gpt():
    import os
    queries = read_jsonl("../sichuan-jike.jsonl")
    results = []
    references_list = []
    url = "http://10.108.1.254:18008/omega-gpt/api/v1/chat/completions"
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl9hcHBfaWQiOiJnbG9iYWwiLCJzdWIiOiJJbnNwdXItQXV0aC1NYW5hZ2VyIiwibG9naW5fdWlkIjoiMTgxMzc4ODU2MTg0OTg4NDY3NCIsImxvZ2luX2FjY291bnRfaWQiOiIxODEzNzg4NTYxOTI5NTc2NDUwIiwiaXNzIjoiSW5zcHVyIiwibG9naW5fbG9naW5uYW1lIjoiYWRtaW4iLCJjbGllbnRfaXAiOiIxNzIuMjAuMC43IiwibG9naW5fYWNjb3VudF9uYW1lIjoi566h55CG5ZGYIiwidXNlcnNfYXBwX2lkIjoiZ2xvYmFsIiwianRpIjoiMS4wIiwibG9naW5fdW5hbWUiOiLnrqHnkIblkZgiLCJleHAiOjE3MjM3ODg1MzQsIm5iZiI6MTcyMzcwMjEzNH0.-_R2Sd1lznA5ABI1wDPjhhQ5-E4JMCw7UfxqEkiRiMBSNoKSLTb7RTMBFl3F4GGonUaZhBK5FXr_1v4193ftgg',
        'Content-Type': 'application/json'
    }
    for query in queries:
        data = {
            "select_param": "sichuan_markdown",
            "chat_mode": "chat_knowledge",
            "model_name": "proxyllm",
            "user_input": query["query"],
            "conv_uid": "f59331a0-5acc-11ef-bc82-0242ac140009"
        }
        response = requests.post(url,
                                 json=data,
                                 headers=headers
                                 )
        content = response.content.decode('utf-8').split("\n\ndata:")[-1]
        output, references = content.split("<references")
        source_list = list(
            set([os.path.basename(quote.split(":")[1]) for quote in references.replace("}", "{").split("{") if
                 "'source'" in quote]))
        qutes = ""
        for source in source_list:
            qutes += source + ","
        qutes = qutes[:-1]
        print(query["id"], "---->\n", output)
        results.append(output)
        references_list.append(qutes)
        # break
    print("开始保存omega-gpt结果。")

    # save_answers_test([queries[0]], results, references_list, path="./answers_omega-gpt.jsonl")
    save_answers_test(queries, results, references_list, path="./answers_omega-gpt-20240815-2.jsonl")
    print("omega-gpt save completed!")


# fastgpt
def fast_gpt():
    queries = read_jsonl("../sichuan-jike.jsonl")
    results = []
    references_list = []
    for query in queries:
        data = {
            "messages": [
                {
                    "dataId": "toHxOGXI69vcKZCuz6jPZ7RI",
                    "role": "user",
                    "content": query["query"]
                }
            ],
            "nodes": [
                {
                    "nodeId": "userGuide",
                    "name": "系统配置",
                    "intro": "可以配置应用的系统参数",
                    "flowNodeType": "userGuide",
                    "isEntry": True,
                    "inputs": [],
                    "outputs": []
                },
                {
                    "nodeId": "workflowStartNodeId",
                    "name": "流程开始",
                    "avatar": "/imgs/workflow/userChatInput.svg",
                    "intro": "",
                    "flowNodeType": "workflowStart",
                    "isEntry": True,
                    "inputs": [
                        {
                            "key": "userChatInput",
                            "renderTypeList": [
                                "reference",
                                "textarea"
                            ],
                            "valueType": "string",
                            "label": "用户问题",
                            "required": True,
                            "toolDescription": "用户问题"
                        }
                    ],
                    "outputs": [
                        {
                            "id": "userChatInput",
                            "key": "userChatInput",
                            "label": "core.module.input.label.user question",
                            "valueType": "string",
                            "type": "static"
                        }
                    ]
                },
                {
                    "nodeId": "7BdojPlukIQw",
                    "name": "AI 对话",
                    "avatar": "/imgs/workflow/AI.png",
                    "intro": "AI 大模型对话",
                    "flowNodeType": "chatNode",
                    "showStatus": True,
                    "isEntry": False,
                    "inputs": [
                        {
                            "key": "model",
                            "renderTypeList": [
                                "settingLLMModel",
                                "reference"
                            ],
                            "label": "core.module.input.label.aiModel",
                            "valueType": "string",
                            "value": "gpt-3.5-turbo"
                        },
                        {
                            "key": "temperature",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "value": 0,
                            "valueType": "number",
                            "min": 0,
                            "max": 10,
                            "step": 1
                        },
                        {
                            "key": "maxToken",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "value": 2000,
                            "valueType": "number",
                            "min": 100,
                            "max": 4000,
                            "step": 50
                        },
                        {
                            "key": "isResponseAnswerText",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "value": True,
                            "valueType": "boolean"
                        },
                        {
                            "key": "quoteTemplate",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "string"
                        },
                        {
                            "key": "quotePrompt",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "string"
                        },
                        {
                            "key": "systemPrompt",
                            "renderTypeList": [
                                "textarea",
                                "reference"
                            ],
                            "max": 3000,
                            "valueType": "string",
                            "label": "core.ai.Prompt",
                            "description": "core.app.tip.chatNodeSystemPromptTip",
                            "placeholder": "core.app.tip.chatNodeSystemPromptTip",
                            "value": ""
                        },
                        {
                            "key": "history",
                            "renderTypeList": [
                                "numberInput",
                                "reference"
                            ],
                            "valueType": "chatHistory",
                            "label": "core.module.input.label.chat history",
                            "required": True,
                            "min": 0,
                            "max": 30,
                            "value": 6
                        },
                        {
                            "key": "userChatInput",
                            "renderTypeList": [
                                "reference",
                                "textarea"
                            ],
                            "valueType": "string",
                            "label": "用户问题",
                            "required": True,
                            "toolDescription": "用户问题",
                            "value": [
                                "workflowStartNodeId",
                                "userChatInput"
                            ]
                        },
                        {
                            "key": "quoteQA",
                            "renderTypeList": [
                                "settingDatasetQuotePrompt"
                            ],
                            "label": "",
                            "debugLabel": "知识库引用",
                            "description": "",
                            "valueType": "datasetQuote",
                            "value": [
                                "iKBoX2vIzETU",
                                "quoteQA"
                            ]
                        }
                    ],
                    "outputs": [
                        {
                            "id": "history",
                            "key": "history",
                            "label": "core.module.output.label.New context",
                            "description": "core.module.output.description.New context",
                            "valueType": "chatHistory",
                            "type": "static"
                        },
                        {
                            "id": "answerText",
                            "key": "answerText",
                            "label": "core.module.output.label.Ai response content",
                            "description": "core.module.output.description.Ai response content",
                            "valueType": "string",
                            "type": "static"
                        }
                    ]
                },
                {
                    "nodeId": "iKBoX2vIzETU",
                    "name": "知识库搜索",
                    "avatar": "/imgs/workflow/db.png",
                    "intro": "调用“语义检索”和“全文检索”能力，从“知识库”中查找可能与问题相关的参考内容",
                    "flowNodeType": "datasetSearchNode",
                    "showStatus": True,
                    "isEntry": False,
                    "inputs": [
                        {
                            "key": "datasets",
                            "renderTypeList": [
                                "selectDataset",
                                "reference"
                            ],
                            "label": "core.module.input.label.Select dataset",
                            "value": [
                                {
                                    "datasetId": "66bd6be93962ba8c3633d4e2"
                                }
                            ],
                            "valueType": "selectDataset",
                            "list": [],
                            "required": True
                        },
                        {
                            "key": "similarity",
                            "renderTypeList": [
                                "selectDatasetParamsModal"
                            ],
                            "label": "",
                            "value": 0.4,
                            "valueType": "number"
                        },
                        {
                            "key": "limit",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "value": 1500,
                            "valueType": "number"
                        },
                        {
                            "key": "searchMode",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "string",
                            "value": "embedding"
                        },
                        {
                            "key": "usingReRank",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "boolean",
                            "value": False
                        },
                        {
                            "key": "datasetSearchUsingExtensionQuery",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "boolean",
                            "value": False
                        },
                        {
                            "key": "datasetSearchExtensionModel",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "string"
                        },
                        {
                            "key": "datasetSearchExtensionBg",
                            "renderTypeList": [
                                "hidden"
                            ],
                            "label": "",
                            "valueType": "string",
                            "value": ""
                        },
                        {
                            "key": "userChatInput",
                            "renderTypeList": [
                                "reference",
                                "textarea"
                            ],
                            "valueType": "string",
                            "label": "用户问题",
                            "required": True,
                            "toolDescription": "需要检索的内容",
                            "value": [
                                "workflowStartNodeId",
                                "userChatInput"
                            ]
                        }
                    ],
                    "outputs": [
                        {
                            "id": "quoteQA",
                            "key": "quoteQA",
                            "label": "core.module.Dataset quote.label",
                            "type": "static",
                            "valueType": "datasetQuote"
                        }
                    ]
                }
            ],
            "edges": [
                {
                    "source": "workflowStartNodeId",
                    "target": "iKBoX2vIzETU",
                    "sourceHandle": "workflowStartNodeId-source-right",
                    "targetHandle": "iKBoX2vIzETU-target-left",
                    "status": "waiting"
                },
                {
                    "source": "iKBoX2vIzETU",
                    "target": "7BdojPlukIQw",
                    "sourceHandle": "iKBoX2vIzETU-source-right",
                    "targetHandle": "7BdojPlukIQw-target-left",
                    "status": "waiting"
                }
            ],
            "variables": {
                "cTime": "2024-08-15 11:11:54 Thursday"
            },
            "appId": "66a9b1fd018d05acf9627a70",
            "appName": "调试-test",
            "detail": True,
            "stream": True
        }

        header = {
            "content-type": "application/json",
            "cookie": "fastgpt_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2Njk3M2ZmNTA3OWUyNmI0YzYyOGFhOTUiLCJ0ZWFtSWQiOiI2Njk3M2ZmNTA3OWUyNmI0YzYyOGFhYjMiLCJ0bWJJZCI6IjY2OTczZmY1MDc5ZTI2YjRjNjI4YWFiZiIsImV4cCI6MTcyMzcxMjA5MSwiaWF0IjoxNzIzMTA3MjkxfQ.qM-D3Dih3NuIInqgsbR8vAraGjovuMPzT5rR7r9DM0w"
        }
        response = requests.post(url="http://10.108.1.254:18012/api/core/chat/chatTest", json=data, headers=header)

        content = response.content.decode('utf-8')
        answer = content.split("event: answer")
        _output = [i.split("data:")[1] for i in answer if "data:" in i and "content" in i and "assistant" in i]
        output = ""
        for i in _output:
            i = json.loads(i)
            output += i["choices"][0]["delta"]["content"]

        _references = answer[-1].split("sourceName")
        references = [reference.split(',"score"')[0][3:-1] for reference in _references if (
                '":"' in reference and ',"score":[{"type":"embedding","value":' in reference and ',"index":' in reference)]
        source_list = list(set(references))
        qutes = ""
        for source in source_list:
            qutes += source + ","
        qutes = qutes[:-1]

        print(query["id"], "---->\n", output)

        results.append(output)
        references_list.append(qutes)
        # break

    print("开始保存fastgpt结果。")

    # save_answers_test([queries[0]], results, references_list, path="./answers_fast-gpt-20240815.jsonl")
    save_answers_test(queries, results, references_list, path="./answers_fast-gpt-20240815-2.jsonl")
    print("fast-gpt save completed!")


if __name__ == '__main__':
    llama_index()
    # omega_gpt()
    # fast_gpt()
