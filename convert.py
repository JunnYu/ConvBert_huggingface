from convbert_huggingface.convert_convbert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if __name__ == "__main__":
    for model_size in ["small", "medium-small", "base"]:
        tf_checkpoint_path = f"E:/github/ConvBert-master/weights/convbert_models/convbert_{model_size}"
        config_file = f"./weights/convbert_{model_size}/config.json"
        pytorch_dump_path = f"./weights/convbert_{model_size}/pytorch_model.bin"
        convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file,
                                         pytorch_dump_path)
