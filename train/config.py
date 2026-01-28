import os

class CFG:
    # model params
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
    load_in_4bit = True
    load_in_8bit = False
    use_gradient_checkpointing = 'unsloth'
    local_files_only=True
    save_dir = 'Llama-3.2-11B-Vision-Instruct'
    last_pth_save_path = os.path.join(save_dir, "last")

    # lora params
    finetune_vision_layers     = True
    finetune_language_layers   = True
    finetune_attention_modules = True
    finetune_mlp_modules       = True
    r = 16
    lora_alpha = 16
    lora_dropout = 0
    bias = "none"
    random_state = 3407
    use_rslora = False
    loftq_config = None

    # data path
    train_data_labels = "train.txt"
    train_data_images = "train"
    train_data_cot = "cot_dataset.json"

    # training params
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    warmup_steps = 50
    num_train_epochs = 2
    learning_rate = 2e-4
    logging_steps = 50
    optim = "adamw_8bit" # adamw_8bit,adamw_torch_fused
    weight_decay = 0.01
    lr_scheduler_type = "linear"
    seed = 3407
    report_to = "none" # Wandb
    save_strategy="epoch"
    dataset_num_proc = 8
    max_seq_length = 2048