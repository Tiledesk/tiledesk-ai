import json
import numpy as np
import re

import gc
from collections import OrderedDict
import time
import importlib
import logging


from tileai.core.abstract_tiletrainer import TileTrainer
from tileai.core.preprocessing.textprocessing import prepare_dataset, save_model, load_model
from tileai.shared import const

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv


logger = logging.getLogger(__name__)

#EmbeddingClassifier 

class TileTrainerSemantic(TileTrainer):

    """
    .environ
    """    

    def __init__(self, language, pipeline, parameters, model):
        import os
        from dotenv import load_dotenv
        load_dotenv('.environ')
        print(f"pipeline: {pipeline}, parameters: {parameters}")
        gptkey = parameters.get("gptkey")
        try: 
            if pipeline[2]=="text-embedding-3-large":
                self.modelname = pipeline[2]
                self.emb_dimension = 3072
            elif pipeline[2]=="text-embedding-3-small":
                self.modelname = pipeline[2]
                self.emb_dimension = 1536
            else:
                self.modelname = "text-embedding-ada-002"
                self.emb_dimension = 1536
        except IndexError:
            self.modelname = "text-embedding-ada-002"
            self.emb_dimension = 1536

        self.language=language
        self.pipeline=pipeline
        self.parameters=parameters
        self.model = model #mode dell'indice su pinecone o chroma
        self.embeddings = OpenAIEmbeddings(api_key=gptkey, model=self.modelname) #default text-embedding-ada-002 1536, text-embedding-3-large 3072, text-embedding-3-small 1536 
        self.index_name = "semantic-sim"
        self.vector_store = self.create_index(index_name=self.index_name, embeddings=self.embeddings, emb_dimension= self.emb_dimension)
        
    
    
    def train(self, dataframe, entities_list, intents_list, synonym_dict):
        
        from langchain_core.documents import Document
        from pprint import pprint
        import pinecone
        
        sentences = dict(
            sentence=[],
            entities=[],
            intent=[]
        )

        mydocs = [] 
        for _, row in dataframe.iterrows():

            #sentences["sentence"].append(row["example"])
            #sentences["entities"].append(row["entities"])
            #sentences["intent"].append(row["intent"])
            document = Document(page_content=row["example"], metadata={"intent":row["intent"]})
            mydocs.append(document)
        

        pprint(mydocs)
        pc = pinecone.Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        ) #, environment=os.environ.get('PINECONE_ENV')
        
        try:
            self.vector_store.delete(delete_all = True, namespace=self.model)
        except Exception as ex:
            print(ex) 
            pass
        
        self.vector_store.from_documents(mydocs, embedding=self.embeddings,index_name=self.index_name,namespace=self.model )

        configuration = {}
        configuration["language"] = self.language
        configuration["pipeline"] = self.pipeline
        configuration["entities"] = entities_list
        configuration["intents"] = intents_list
        configuration["synonim"] = synonym_dict
        configuration["model"] = self.model
        
        config_label_json = self.model+"/"+const.MODEL_CONFIG
        print(config_label_json)
    
        with open(config_label_json, 'w', encoding='utf-8') as f:
            json.dump(configuration, f, ensure_ascii=False, indent=4)
        
        return None
    

   


    def query(self, configuration, query_text):
        
            
        from pprint import pprint
        from langchain_core.documents import Document
        #do = Document()
        #do
        
        docs = self.vector_store.similarity_search_with_relevance_scores(query=query_text,namespace=model_classifier,k=6)
        pprint(docs)
        

        intent_r = []
    
        # TODO qui non ottengo una lista di intent e confidence, 
        # bisogna verificare la se con la query si riesce a fare qualcosa di meglio.
        # possibile soluzione prendere 1 documento. Che è il più simile e per ogni intent fare una query. 
        
        results_dict = {}
        results_dict["text"]= query_text
        results_dict["intent"]={"name":docs[0][0].metadata.get("intent"), 
                                "confidence": docs[0][1] }
            
        results_dict["intent_ranking"] = [{"name":name.metadata.get("intent"),"confidence":confidence} for name,confidence in docs ]
            

       
        return docs[0][0].metadata,  results_dict



    def query_http(self, vocabll,model_classifier, id2label,tokenizer, query_text):
        from pprint import pprint
        from langchain_core.documents import Document
        #do = Document()
        #do
        
        docs = self.vector_store.similarity_search_with_relevance_scores(query=query_text,namespace=model_classifier,k=10)
        pprint(docs)
        

        intent_r = []
    
        # TODO qui non ottengo una lista di intent e confidence, 
        # bisogna verificare la se con la query si riesce a fare qualcosa di meglio.
        # possibile soluzione prendere 1 documento. Che è il più simile e per ogni intent fare una query. 
        
        results_dict = {}
        results_dict["text"]= query_text
        results_dict["intent"]={"name":docs[0][0].metadata.get("intent"), 
                                "confidence": docs[0][1] }
            
        results_dict["intent_ranking"] = [{"name":name.metadata.get("intent"),"confidence":confidence} for name,confidence in docs ]
            

       
        return docs[0][0].metadata,  results_dict
        
    
    def delete_pinecone_index(self,index_name='all'):
        from pinecone import Pinecone

        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        )#, environment=os.environ.get('PINECONE_ENV')
        

        if index_name == 'all':
            indexes = pc.list_indexes().names()
            print('Deleting all indexes ... ')
            for i in indexes:
                pc.delete_index(i)
            print('Ok')
        else:
            print(f'Deleting index {index_name} ...', end='')
            pc.delete_index(index_name)
            print('Ok')

    def create_index(self,index_name,embeddings, emb_dimension):
     
        import pinecone
        from langchain_community.vectorstores import Pinecone

       
       
        #pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get('PINECONE_ENV'))
        pc = pinecone.Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        ) #, environment=os.environ.get('PINECONE_ENV')

        
        if index_name in pc.list_indexes().names():
            print(f'Index {index_name} esiste. Loading embeddings ... ', end='')
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
        #    vector_store.from_documents(chunks, embeddings)
            print('Ok')
        else:
            print(f'Creazione di index {index_name} e embeddings ...', end='')
            pc.create_index(index_name, dimension=emb_dimension, metric='cosine',spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-west-2") 
            )
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
        #  vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
            print('Ok')

        return vector_store



