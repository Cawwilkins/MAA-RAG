from sentence_transformers import CrossEncoder

model_name = "BAAI/bge-reranker-base"
local_dir = r".\models\bge-reranker-base"

model = CrossEncoder(model_name)
model.save(local_dir)

print("Saved reranker locally.")