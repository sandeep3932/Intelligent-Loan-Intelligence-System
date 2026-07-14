import pickle
import json
import os
from gensim.models import Word2Vec

def dump_pickle_info():
    pkl_path = "saved_models/one_hot_encoder.pkl"
    out_path = "saved_models/one_hot_encoder_info.txt"
    if not os.path.exists(pkl_path):
        return
    try:
        with open(pkl_path, "rb") as f:
            encoder = pickle.load(f)
    except Exception as e:
        with open(out_path, "w") as f:
            f.write(f"Could not read pickle file: {e}\n")
        return
    with open(out_path, "w") as f:
        f.write("=== One Hot Encoder Info ===\n")
        f.write(f"Type: {type(encoder)}\n")
        if hasattr(encoder, "categories_"):
            f.write(f"Categories: {encoder.categories_}\n")
        if hasattr(encoder, "drop_idx_"):
            f.write(f"Drop Index: {encoder.drop_idx_}\n")
        if hasattr(encoder, "get_feature_names_out"):
            f.write(f"Feature Names: {encoder.get_feature_names_out()}\n")

def dump_w2v_info(model_name):
    model_path = f"saved_models/{model_name}.model"
    out_path = f"saved_models/{model_name}_info.txt"
    if not os.path.exists(model_path):
        return
    
    model = Word2Vec.load(model_path)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} Info ===\n")
        f.write(f"Vector size: {model.vector_size}\n")
        f.write(f"Vocabulary size: {len(model.wv)}\n\n")
        f.write("Vocabulary words (first 100):\n")
        words = list(model.wv.index_to_key)
        f.write(str(words[:100]) + "\n\n")
        
        if len(words) > 0:
            sample_word = words[0]
            f.write(f"Sample vector for word '{sample_word}':\n")
            f.write(str(model.wv[sample_word]) + "\n")

if __name__ == "__main__":
    dump_pickle_info()
    dump_w2v_info("w2v_cbow")
    dump_w2v_info("w2v_sg")
    print("Readable text files have been created in the saved_models folder!")
