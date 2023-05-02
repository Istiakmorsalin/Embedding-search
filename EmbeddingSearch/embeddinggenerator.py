from deepface import DeepFace
from deepface.commons import functions, distance
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image

class EmbeddingGenerator:

    def generateEmbedding(self, image_path): 
        embedding_objs = DeepFace.represent(img_path = image_path)
        embedding = embedding_objs[0]["embedding"]
        embedding_numpy = np.array(embedding)
        print(embedding_numpy.shape)
        embedding_string = ",".join(str(element) for element in embedding)
        embedding_list = embedding_string.split(',')
        print(embedding_list)

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def generate_img2vec_dict(self, df,image_path, batch_size=500):
        output_dict={}

        for batch in self.chunker(df, batch_size):
         image_filenames=batch['path'].values.tolist()
         images=[]
         converted=[]
        
        for img_fn in image_filenames:
            try:
                img = Image.open(image_path + img_fn)
                images.append(img)
                converted.append(img_fn)
            except:
                #unable_to_convert -> skip to the next image
                continue
        
        #Generate vectors for all images in this batch
        vec_list = img2vec.get_vec(images)
        
        #update the dictionary to be returned
        batch_dict= dict(zip(converted, vec_list))
        output_dict.update(batch_dict)
        
    
        return output_dict

