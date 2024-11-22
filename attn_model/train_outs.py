import sys
import subprocess

outs = [1,2,3,4,5]#2,3,4,5
seq_lens = [300,300,300,300,300]

model_params = "./model_params.yml"
mhn_batch_size = 4
mhn_epochs = 100

arch_no = 0
for out in range(len(outs)):

    arch_no += 1
    print(f"Training Arch No. {arch_no}")    
    print(f"Running Model {outs[out]} with sequence length {seq_lens[out]}")
    command = (f"CUDA_VISIBLE_DEVICES={0} python main.py --yaml_file {model_params}"  
                f" --out {outs[out]} --seq_len {seq_lens[out]} --mhn_batch_size {mhn_batch_size}"
                f" --mhn_epochs {mhn_epochs}"
            )

    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    last_lines = res.stdout.splitlines()[-15:] # print last 8 lines from the terminal
    decoded_data = [item.decode('utf-8') for item in last_lines]

    formatted_output = "\n".join(decoded_data)

    print(formatted_output)