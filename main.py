
import os
# os.environ["PROMPT_INDEX"]="0"

import requests
from elasticsearch import Elasticsearch
# from app import prompt_config
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import logging

es=Elasticsearch([{"host":"169.48.177.59","port":9200,"scheme":"http"}])

print(es.info()['version']['number'])

Default_size=2
API_TOKEN_IBM="FWjEm_bPc24heXJ3mFHwfsM4jhE3JN66Ag6NMki80KTK"
PROJECT_ID_IBM="41d71924-826e-4873-a7d3-5a16d198e6f6"

print(es.ping())
model_id=".elser_model_2_linux-x86_64_search"
def Search_Docs_gpt(query, username,path):
        response = es.search(
            index="teamsyncfirstn",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"username": username}},# Filter by username
                            {"match_phrase_prefix": {"path": path}}, 
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "sparse_vector": {
                                                "field": "text_embedding",
                                                "inference_id": model_id,
                                                "query": query
                                            }
                                        },
                                        {
                                            "match": {
                                                "text": query  # Match exact phrase in the text field
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                "_source": [
                    "text", "pageNo", "fId", "username", "tables", "fileName"
                ]  
            }
        )
        return response['hits']['hits']

def search_documents_gpt(query_text, user_name, model_type, answerType, path):
    hits = Search_Docs_gpt(query_text, user_name, path)
    filelist = []
    search_results = []
    lst = []

    if answerType not in ["singleDocument", "multiDocument"]:
        logging.warning(f"Unsupported answer type : {answerType}")
        return [{"text": f"Unsupported answer type: {answerType}. Supported types: ['singleDocument', 'multiDocument']"}]

    if model_type not in ["mistral", "phi3"]:
        logging.warning(f"Unsupported model type : {model_type}")
        return [{"text": f"Unsupported model type: {model_type}. Supported types: ['mistral', 'phi3']"}]
    
    if not hits:
        logging.warning(f"No documents found -->")
        return [{"text": "No documents found for the query."}]

    file_id = hits[0]["_source"].get("fId", "")
    page_no = hits[0]["_source"].get("pageNo", "")
    text = hits[0]["_source"].get("text", "")
    logging.warning(f"text1 (hits[0][_source][text]) --> {text}")
    combined_text_single_doc = above_and_below_pagedata(text, int(page_no), file_id)
    combined_text_multi_doc = ""

    for hit in hits:
        score = hit["_score"]
        if score > 3:
            filename = hit["_source"].get("fileName", "")
            if filename not in filelist:
                file_id = hit["_source"].get("fId", "")
                text = hit["_source"].get("text", "")
                page_no = hit["_source"].get("pageNo", "")
                base, extension = os.path.splitext(filename)
                extension = str.upper(extension)
                if extension != '.CSV':
                    combined_text_multi_doc += "\n" + text
                table_data = hit["_source"].get("tables", "")
                filelist.append(filename)
                lst.append(table_data)
                search_results.append({"filename": filename, "fId": file_id, "page_no": page_no, "score": score})
                # es_id = review_data.add_document(query_text,text,"None",user_name,score,file_id)
                # logging.warning(f"Review Data Document ID--> {es_id}")

    if not search_results:
        logging.warning(f"Search Results --> []")
        return [{"text": "I am unable to provide an answer based on the information I have."}]

    if answerType == "singleDocument":
        if model_type == "mistral":
            model_answer = ibm_cloud_granite(combined_text_single_doc, query_text)
        
    elif answerType == "multiDocument":
        if model_type == "mistral":
            model_answer = ibm_cloud_granite(combined_text_multi_doc, query_text)
    


    search_results.insert(0, {"text": model_answer})
    logging.warning(f"search_results :: {search_results}")
    return search_results

def Data_By_FID_ES(f_id, query,size=Default_size):
        response = es.search(
            index="teamsyncfirstn",
            body={
                "size":size,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"fId": f_id}},  # Filter by fid
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "sparse_vector": {
                                                "field": "text_embedding",
                                                "inference_id": model_id,
                                                "query": query
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        )

        if response['hits']['hits']:
            return response['hits']['hits']
def Data_By_pageno(page_no, fid):
        match = es.search(
            index='teamsyncfirstn',
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"pageNo": page_no}},
                            {"match": {"fId": fid}}
                        ]
                    }
                },
                "script_fields": {
                    "text": {
                        "script": "params['_source']['text']"
                    }
                }
            }
        )
        if match['hits']['hits']:
            details = match['hits']['hits']
            return details[0]["fields"]

def above_and_below_pagedata(text, page_no, file_id):
    page_no_below = page_no + 1
    below_page_text = Data_By_pageno(page_no_below, file_id)

    if below_page_text is not None:
        below_page_text = below_page_text['text'][0]
    else:
        below_page_text = ''
    if (page_no != 1):
        page_no_above = page_no - 1
        above_page_text = Data_By_pageno(page_no_above, file_id)

        if above_page_text is not None:
            above_page_text = above_page_text['text'][0]
        else:
            above_page_text = ''
        return above_page_text + text + below_page_text

    else:
        page_no_above = page_no + 2
        below_page_text_2 = Data_By_pageno(page_no_above, file_id)

        if below_page_text_2 is not None:
            below_page_text_2 = below_page_text_2['text'][0]
        else:
            below_page_text_2 = ''
        return text + below_page_text + below_page_text_2
def ibm_cloud_granite(text,query):

    authenticator = IAMAuthenticator(API_TOKEN_IBM)
    service = "Bearer " + authenticator.token_manager.get_token()
    
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    prompt = prompt_config.get_prompt(text=text, query=query)
    # prompt=truncate_text(prompt,7500)
    body = {
        "input":prompt ,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "stop_sequences": [],
            "repetition_penalty": 1
        },
        # "model_id": "ibm/granite-13b-chat-v2",
        "model_id": "meta-llama/llama-3-1-8b-instruct",
        "project_id": PROJECT_ID_IBM,
        "moderations": {
            "hap": {
                "input": {"enabled": False},  # Disable input moderation
                "output": {"enabled": False}  # Disable output moderation
            }
        }
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": service
    }

    response = requests.post(
        url,
        verify=False,
        headers=headers,
        json=body
    )
    
    if response.status_code != 200:
        logging.warning(f"Status Code --> {response.status_code}")
        return ("model context window exceeded for this document"+str(response.text))
    data = response.json()
   
    return data['results'][0]['generated_text']


   
def Data_By_FID(fid, query, model_type):
    hits = Data_By_FID_ES(fid, query)
    if Default_size!=1:
        combined_text=""
        for i in hits:
            try:
                text =i["_source"].get("text", "")
            except Exception as e:
                return [{"text": "No hits from database"}]

            page_no = hits[0]["_source"].get("pageNo", "")
            combined_text = combined_text +"\n" +above_and_below_pagedata(text, int(page_no), fid)
        logging.warning(f"combined_text --> {combined_text}")
        if model_type == "mistral":
            model_answer = ibm_cloud_granite(combined_text, query)

        else:
            outres = f"model type not match :: {model_type}, modeltype :: ['mistral','phi3']"
            logging.warning(f"{outres}")
            return outres

        logging.warning(f"model_answer --> {model_answer}")
        return [{"text": model_answer}]
    else:
        print(f"entered in size {Default_size}")
        try:
            text = hits[0]["_source"].get("text", "")
        except Exception as e:
            return [{"text": "No hits from database"}]

        page_no = hits[0]["_source"].get("pageNo", "")
        combined_text = above_and_below_pagedata(text, int(page_no), fid)
        logging.warning(f"combined_text --> {combined_text}")
        if model_type == "mistral":
            model_answer = ibm_cloud_granite(combined_text, query)

        else:
            outres = f"model type not match :: {model_type}, modeltype :: ['mistral','phi3']"
            logging.warning(f"{outres}")
            return outres

        logging.warning(f"model_answer --> {model_answer}")
        return [{"text": model_answer}]


if __name__=="__main__":
    fid="67bee2bfde7618390f109558"
    query="full form OF RFP"
    answer=Data_By_FID(fid,query,model_type="mistral")
    print(answer)

    # result=search_documents_gpt(query,"amit_test.com","mistral","multiDocument",path="T_")
    # print("results from search documents",result)
