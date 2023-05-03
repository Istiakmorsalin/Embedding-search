import pandas as pd
from redisoperations import RedisConnector
from deepface import DeepFace
from deepface.commons import functions, distance
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from embeddinggenerator import EmbeddingGenerator
from redis import Redis


class DataLoader:
    NUMBER_PRODUCTS = 100000

    def __init__(self):
        pass

    def loadData(self): 
        print("Load Amazon Product and Image metadata")
        # Load Product data and truncate long text fields
        all_prods_df = pd.read_csv('data/product_image_data.csv')
        all_prods_df['primary_key'] = all_prods_df['item_id'] + '-' + all_prods_df['domain_name']
        #print(all_prods_df.shape)
        #print(all_prods_df.head(5))
        subset_df = all_prods_df.head(self.NUMBER_PRODUCTS)
        #print(subset_df.iloc[0])
        return all_prods_df

    def build_index(self):
        PRODUCT_IMAGE_VECTOR_FIELD='product_image_vector'
        IMAGE_VECTOR_DIMENSION=512
        
        redis_conn = RedisConnector().connect()

        #flush all data
        #redis_conn.flushall()

        #load data
        product_metadata = self.loadData()

        #generate image vectors
        image_path = 'data/images/'
       # img2vec = Img2Vec(cuda=False)
        img2vec_dict = EmbeddingGenerator().generate_img2vec_dict(product_metadata,image_path)
        print("Anik")
        print(img2vec_dict)

        #create flat index & load vectors
        #self.create_flat_index(redis_conn, PRODUCT_IMAGE_VECTOR_FIELD, self.NUMBER_PRODUCTS, IMAGE_VECTOR_DIMENSION,'COSINE')
       # self.load_vectors(redis_conn, product_metadata, img2vec_dict, PRODUCT_IMAGE_VECTOR_FIELD)

        
    def create_flat_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2'):
        redis_conn.ft().create_index([
        
        VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }),
            TagField("product_type"),
            TextField("item_name"),
            TagField("country")        
    ])
        
    def create_hnsw_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2',M=40,EF=200):
        redis_conn.ft().create_index([

        VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, "EF_CONSTRUCTION": EF}),
            TagField("product_type"),
            TextField("item_name"),
            TagField("country")     
    ]) 

    def load_vectors(client:Redis, product_metadata, vector_dict, vector_field_name):
        p = client.pipeline(transaction=False)

        for index in product_metadata.keys():    
        #hash key
            key='product:'+ product_metadata[index]['primary_key']
        
        #hash values
            item_metadata = product_metadata[index]
            item_path = item_metadata['path']
        
        if item_path in vector_dict:
            #retrieve vector for product image 
            product_image_vector = vector_dict[item_path].astype(np.float32).tobytes()
            item_metadata[vector_field_name]=product_image_vector
            
            # HSET
            #p.hset(key,mapping=product_data_values)
            p.hset(key,mapping=item_metadata)
            
        p.execute()       


