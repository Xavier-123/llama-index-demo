import json

import requests
from demo.pipeline.qa import read_jsonl, save_answers_test
import os


queries = read_jsonl("../question-pdd.jsonl")



# llama_index
def llama_index():
    results, references_list = [], []
    url = "http://127.0.0.1:9111/chaxun"
    for query in queries:
        data = {
            "query": query["query"],
            "reranker_top": 2
        }
        response = requests.post('http://127.0.0.1:9111/chaxun',
                                 json=data)
        content = response.content.decode('utf-8')
        content = json.loads(content)
        output = content["res"]["text"]
        references = content["res"]["quote"]

        results.append(output)
        references_list.append(references)
        break

    save_answers_test(queries, results, references_list, path="output/answers_fast-gpt.jsonl")



# omega-gpt
def omega_gpt():
    import os
    results = []
    references_list = []
    url = "http://192.168.12.188:32777/omega-gpt/api/v1/chat/completions"
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl9hcHBfaWQiOiIxNjgxMTM4OTM2OTUyMzExODA5Iiwic3ViIjoiSW5zcHVyLUF1dGgtTWFuYWdlciIsImxvZ2luX3VpZCI6IjE2OTY0MjE2NzM0NjQ2NzIyNTgiLCJsb2dpbl9hY2NvdW50X2lkIjoiMTY0MTY0NzU0Mjc1NDQ3MTkzOCIsImlzcyI6Ikluc3B1ciIsImxvZ2luX2xvZ2lubmFtZSI6ImlhaS1hZG1pbiIsImNsaWVudF9pcCI6IjE3Mi4xMC4yNS4xOCIsImxvZ2luX2FjY291bnRfbmFtZSI6IkFJ5bmz5Y-w566h55CG5ZGYIiwidXNlcnNfYXBwX2lkIjoiZ2xvYmFsIiwianRpIjoiMS4wIiwibG9naW5fdW5hbWUiOiJBSeW5s-WPsOeuoeeQhuWRmCIsImV4cCI6MTcyMjMwNTI0MiwibmJmIjoxNzIyMjE4ODQyfQ.Kp-ygyGGC9EEBSD6EaOVHfLsnFLCBr1cv5e397dauYIjtFfesrXsY6ZEHG6KWHuJcmcqdYhtWiCJ5Cgj1oTFyQ',
        'Content-Type': 'application/json'
    }
    for query in queries:
        data = {
            "select_param": "Product_Department_Data",
            "chat_mode": "chat_knowledge",
            "model_name": "proxyllm",
            "user_input": query["query"],
            "conv_uid": "468bfc62-4d5e-11ef-9739-0ebd84354688"
        }
        response = requests.post(url,
                                 # data=json.dumps(data),
                                 json=data,
                                 headers=headers
                                 )
        # content = '''大模型通常指的是基础大模型，这是由管理员在平台上预置的各类模型，它们具有广泛的适用性和较高的参数量。普通用户可以浏览这些大模型，并基于某个大模型进行微调，以适应特定的任务需求。基础大模型的信息包括名称、简介、参数量以及开源厂商等，用户还可以下载这些模型或查看详细的模型信息。\n\n<references title='References" references="[{&quot;name&quot;: &quot;AI训练平台功能说明文档.docx&quot;, &quot;chunks&quot;: [{&quot;id&quot;: 16131, &quot;content&quot;: &quot;模型部署模型管理模型管理主要用于管理通过实验流程或训练任务保存的模型，用户可以查看模型列表、下载或删除模型文件，方便使用者对不同资源进行复用和调度。模型支持多版本管理，同一个训练任务多次训练生成的不同的模型版本，点击“&gt;”展开查看子版本。新增模型点击新增按钮，可自行导入训练好的模型文件，文件格式要求为zip。迁移训练支持基于模型进行迁移训练，需要使用算法支持迁移训练，训练过程同模板化训练的训练作业模块中新建训练任务（只是基础模型为当前保存的模型）。发布服务支持基于模型和推理镜像发布服务，推理镜像默认选择训练算法对应的推理镜像，发布后用户可以在服务列表进行启动、查看等操作。删除模型用户可以选择不需要的模型，单击右侧的“删除模型”，在弹出的提示框选择“确定”后即可删除模型。Tips：删除操作是不可逆的，请用户谨慎删除。下载模型支持将模型文件下载至本地。查看模型详情支持查看模型基本信息和模型评估指标等信息。基础大模型&quot;, &quot;meta_info&quot;: &quot;{'source': '/app/pilot/data/Product_Department_Data/AI训练平台功能说明文档.docx'}&quot;, &quot;recall_score&quot;: 0.6619983315467834}, {&quot;id&quot;: 16132, &quot;content&quot;: &quot;支持将模型文件下载至本地。查看模型详情支持查看模型基本信息和模型评估指标等信息。基础大模型平台由管理员预置各类基础大模型，普通用户可浏览并基于某大模型进行微调。基础大模型模块内展示大模型的名称、简介、参数量、开源厂商等，支持下载、查看详情等功能。服务管理服务管理可支持用户进行服务发布，提供删除、启停、添加到项目等基础功能操作。新增服务支持两种创建推理服务的方式：模型发布服务：该种方式由训练任务训练出模型之后，由模型发布推理服务，发布时可选择对应的推理镜像。手动新增服务：自行创建一条服务记录，即该种方式需要自行配置推理服务信息及接口信息并上传镜像。其中新增模型发布服务分为新增来源于结构化数据模型的服务和新增来源于非结构化数据模型的服务。新增数据服务基于推理实验创建服务并填写服务镜像的名称，保存后可自动一键打包服务镜像。新增通用服务手动新增服务，自行配置服务、接口、环境变量等信息，可配置多个接口，常用场景为手动上传图像能力镜像并启动服务。上传镜像&quot;, &quot;meta_info&quot;: &quot;{'source': '/app/pilot/data/Product_Department_Data/AI训练平台功能说明文档.docx'}&quot;, &quot;recall_score&quot;: 0.560359001159668}, {&quot;id&quot;: 16130, &quot;content&quot;: &quot;Tips：其中input1指原始数据库表。output 方法：这个方法用于提供输出节点对应的对象。根据传入的索引，如果索引为 0，则返回原始数据帧，否则返回训练好的 kmeans 模型对象。model_name 方法：返回模型的名称，这里是 \&quot;KMeansOutlier\&quot;。feature 方法：返回算子使用的数据字段，它返回了一个名为 features 的变量，这个变量在前文有传参，通过手动选择输入算子所需要使用的字段。label 方法：返回标记字段，但在这里返回的是 None，因为这是一个无监督学习算法，不需要标记字段。上线自定义算子针对编辑好的自定义算子可点击进行算子上线，上线后即可在算子化实验画布中使用，如算子不需要使用则可下线。共享自定义算子用户可将自定义编写好的算子分享至所有人、某用户或租户，共享来源的算子也可在算子化实验画布中使用。模型部署模型管理&quot;, &quot;meta_info&quot;: &quot;{'source': '/app/pilot/data/Product_Department_Data/AI训练平台功能说明文档.docx'}&quot;, &quot;recall_score&quot;: 0.5553447008132935}]}, {&quot;name&quot;: &quot;AI平台部署文档.docx&quot;, &quot;chunks&quot;: [{&quot;id&quot;: 16153, &quot;content&quot;: &quot;repository：镜像所在的仓库名，如果现场的harbor仓库名称和默认值（iai-release）不同，则需要修改tag：镜像版本，镜像名称是固定的，不允许修改，能修改的只有镜像的版本service相关配置：service的配置和模块的IP以及端口有关，service的下级配置通常有：type: type表示service的类型，一般为ClusterIP和NodePort两种取值，ClusterIP表示集群内可访问，外部不可访问；NodePort表示外部也可以访问。nodePort：当type为ClusterIP时，nodePort值为空；当type为NodePort时，需要指定一个用于外部访问的端口，K8s默认允许的端口范围是30000-32767resources相关配置：resources的配置和模块所使用的资源有关，resources的下级配置通常有：1.&quot;, &quot;meta_info&quot;: &quot;{'source': '/app/pilot/data/Product_Department_Data/AI平台部署文档.docx'}&quot;, &quot;recall_score&quot;: 0.5241457223892212}, {&quot;id&quot;: 16154, &quot;content&quot;: &quot;1.requests：表示申请的最小资源，包括CPU和内存，CPU的取值可以是整型数1、2，表示核数，也可以是500m，表示0.5核（1000m=1核）；内存可以写成512Mi，或者2Gi，分别表示512MB和2GB内存。requests值不宜设置过大，如果集群剩余资源不满足requests的要求，模块会一直Pending，等待资源。limits：表示模块申请使用的最大资源数量，取值和requests类似。该资源和K8s集群的资源配额有关，如果命名空间中所有资源的limits总和大于命名空间的配额，模块会启动失败。persistence相关配置：persistence的配置和模块的持久化有关，不同的模块具体的变量名有所差异，但很类似。persistence的下级配置通常有：enabled: 是否启用持久化storageClass: 存储类，需要和7.1.4章节中创建的StorageClass保持一致&quot;, &quot;meta_info&quot;: &quot;{'source': '/app/pilot/data/Product_Department_Data/AI平台部署文档.docx'}&quot;, &quot;recall_score&quot;: 0.5124281048774719}]}]" />'''
        content = response.content.decode('utf-8').split("\n\ndata:")[-1]
        output, references = content.split("<references")
        # output = content.split("<references")
        source_list = list(set([os.path.basename(quote.split(":")[1]) for quote in references.replace("}", "{").split("{") if "'source'" in quote]))
        qutes = ""
        for source in source_list:
            qutes += source + ","
        qutes = qutes[:-1]
        results.append(output)
        references_list.append(qutes)
        break

    save_answers_test(queries, results, references_list, path="output/answers_omega-gpt.jsonl")






# fastgpt
def fast_gpt():
    results = []
    references_list = []
    for query in queries:
        data = {
            "messages": [
                {
                    "dataId": "tJv4lgnsZEsfQAlREaUyt0Sk",
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
                                    "datasetId": "6698c1b30d4d19b8167e8a8f"
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
                "cTime": "2024-07-29 11:10:38 Monday"
            },
            "appId": "6698cd340d4d19b8167eec01",
            "appName": "调试-OMEGA问答调试",
            "detail": True,
            "stream": True
        }

        header = {
            "content-type": "application/json",
            "cookie": "fastgpt_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2Njk3M2ZmNTA3OWUyNmI0YzYyOGFhOTUiLCJ0ZWFtSWQiOiI2Njk3M2ZmNTA3OWUyNmI0YzYyOGFhYjMiLCJ0bWJJZCI6IjY2OTczZmY1MDc5ZTI2YjRjNjI4YWFiZiIsImV4cCI6MTcyMjQwNjI1MywiaWF0IjoxNzIxODAxNDUzfQ.-49AjUSHhuJ2GVM5xUlOLF2ax6yTsoijnk9UwnlA3QQ"
        }
        response = requests.post(url="http://10.108.1.254:18012/api/core/chat/chatTest", json=data, headers=header)
        # print(response)
        # response.content.decode("utf-8")

        content = response.content.decode('utf-8').split("\n\ndata:")[-1]
        output, references = content.split("event: answer")
        # output = content.split("<references")
        source_list = list(
            set([os.path.basename(quote.split(":")[1]) for quote in references.replace("}", "{").split("{") if
                 "'source'" in quote]))
        qutes = ""
        for source in source_list:
            qutes += source + ","
        qutes = qutes[:-1]
        results.append(output)
        references_list.append(qutes)
        break

    save_answers_test(queries, results, references_list, path="output/answers_fast-gpt.jsonl")





if __name__ == '__main__':
    llama_index()
    # fast_gpt()
    # omega_gpt()