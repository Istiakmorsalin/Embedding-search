from deepface import DeepFace
from deepface.commons import functions, distance
import numpy as np
from PIL import Image

class EmbeddingGenerator:
    
    def generateEmbeddings(self, image_paths): 
        embedding_list = []
        for image_path in image_paths:
            try: 
                embedding_objs = DeepFace.represent(img_path = image_path)
                embedding = embedding_objs[0]["embedding"]
                embedding_numpy = np.array(embedding)
                embedding_list.append(embedding_numpy)
            except Exception as e:
                print(e)    
        return embedding_list

        
    def chunker(self,seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def do_chunker_operation(self, df, batch_size=500):
        for batch in self.chunker(df, batch_size):
            image_filenames=batch['path'].values.tolist()
            return image_filenames 

    
    def generate_img2vec_dict(self, df,image_path, batch_size=500):
        output_dict={}

        image_filenames = self.do_chunker_operation(df,batch_size)
        images=[]
        converted=[]
        image_paths=[]
        
        for img_fn in image_filenames:
            try:
                img = Image.open(image_path + img_fn)
                image_paths.append(image_path + img_fn)
                images.append(img)
                converted.append(img_fn)
            except Exception as e:
                print(e)
                continue
        
        #Generate embeddings for all images in this batch
        embedding = self.generateEmbeddings(image_paths)
        
        print(type(embedding)) 
        
        #update the dictionary to be returned
        batch_dict= dict(zip(converted, embedding))
        output_dict.update(batch_dict)
        
        return output_dict
