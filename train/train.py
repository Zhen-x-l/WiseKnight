from utiles import make_think_data_two_versions
from config import CFG
from unsloth import FastVisionModel 
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import torch
def main():
    model, tokenizer = FastVisionModel.from_pretrained(
        CFG.model_name,
        load_in_4bit = CFG.load_in_4bit,
        load_in_8bit = CFG.load_in_8bit,
        use_gradient_checkpointing = CFG.use_gradient_checkpointing, 
        local_files_only=True
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = CFG.finetune_vision_layers, 
        finetune_language_layers   = CFG.finetune_language_layers, 
        finetune_attention_modules = CFG.finetune_attention_modules, 
        finetune_mlp_modules       = CFG.finetune_mlp_modules, 

        r = CFG.r,          
        lora_alpha = CFG.lora_alpha,  
        lora_dropout = CFG.lora_dropout,
        bias = CFG.bias,
        random_state = CFG.random_state,
        use_rslora = CFG.use_rslora,  
        loftq_config = CFG.loftq_config, 
        
    )
    converted_dataset = make_think_data_two_versions(
        label_file=CFG.train_data_labels,
        image_dir=CFG.train_data_images,
        train_data_cot=CFG.train_data_cot,
        disable_marker="<|think|><|think|>",
        disable_output_mode="label_only",
    )
    

    FastVisionModel.for_training(model) # Enable for training!

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = CFG.per_device_train_batch_size,
            gradient_accumulation_steps = CFG.gradient_accumulation_steps,
            warmup_steps = CFG.warmup_steps,
            num_train_epochs = CFG.num_train_epochs, 
            learning_rate = CFG.learning_rate,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = CFG.logging_steps,
            optim = CFG.optim, 
            weight_decay =CFG.weight_decay,
            lr_scheduler_type = CFG.lr_scheduler_type,
            seed = CFG.seed,
            output_dir = CFG.save_dir,
            report_to = CFG.report_to,     
            save_strategy = CFG.save_strategy, 
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = CFG.dataset_num_proc,
            max_seq_length = CFG.max_seq_length,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train(resume_from_checkpoint = None)

    #Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(CFG.last_pth_save_path)
    tokenizer.save_pretrained(CFG.last_pth_save_path)