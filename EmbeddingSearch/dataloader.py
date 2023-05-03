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
        
        return img2vec_dict