import torch
from convbert_huggingface import ConvBertModel, ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("weights/convbert_base")
text = "it is a sunny day, i want to go out!"
x = tokenizer(text, return_tensors="pt")
x_cuda = {}
for n, item in x.items():
    x_cuda[n] = item.cuda()
for model_size in ["small", "medium-small", "base"]:
    model = ConvBertModel.from_pretrained(
        f"weights/convbert_{model_size}").cuda()
    with torch.no_grad():
        pt_out = model(**x_cuda)[0].cpu()
    tf_out = torch.load(f"./tf_{model_size}_output.pt")
    print(f"{model_size} max difference is :",
          (tf_out - pt_out).abs().max())
    print(f"{model_size} mean difference is :",
          (tf_out - pt_out).abs().mean())