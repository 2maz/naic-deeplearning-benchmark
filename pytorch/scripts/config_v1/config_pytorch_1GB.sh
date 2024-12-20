#!/bin/bash

NUM_GPU=${1:-1}
GPU_SIZE=${2:-1}
NUM_EXP=1

function scale() {
    # 1rst argument is base-batchsize for 1GB
    # 2nd argument is an additional scaling introduced for multiple GPUs 
    #  to avoid out-of-memory errors
    SCALING_FACTOR=$(($NUM_GPU*$GPU_SIZE))
    if [ "$NUM_GPU" -gt 1 ]; then
        if [ "$2" != "" ]; then
            export MULTIPLE_GPUS="*$2"
        fi
    fi
    NUMBER=`echo "$1*${SCALING_FACTOR}${MULTIPLE_GPUS}" | bc`
    echo ${NUMBER%.*}
}

PyTorch_SSD_FP32_PARAMS=(
             "benchmark/Detection/SSD"
             args
             --data                   "/data/object_detection"
             --batch-size             "$(scale 3 0.6)"
             --benchmark-warmup       "50"
             --benchmark-iterations   "200"
             --learning-rate          "0"
             --num-workers            "64"
           )

PyTorch_SSD_AMP_PARAMS=(
             "benchmark/Detection/SSD"
             args
             --data                   "/data/object_detection"
             --batch-size             "$(scale 6 0.6)"
             --benchmark-warmup       "50"
             --benchmark-iterations   "200"
             --amp
             --learning-rate          "0"
             --num-workers            "64"
           )

PyTorch_resnet50_FP32_PARAMS=(
             "benchmark/Classification/ConvNets"
             args
                                      "/data/imagenet"
             --arch                   "resnet50"
             --epochs                 "2" 
             --prof                   "100" 
             --batch-size             "$(scale 8 0.7)"
             --raport-file            "benchmark.json"
             --print-freq             "1"
             --training-only
             --data-backend "synthetic"
             --workers                "64"
           )

PyTorch_resnet50_AMP_PARAMS=(
             "benchmark/Classification/ConvNets"
             args
                                      "/data/imagenet"
             --arch                   "resnet50"
             --amp
             --static-loss-scale      "256"
             --epochs                 "2" 
             --prof                   "100" 
             --batch-size             "$(scale 16 0.7)"
             --raport-file            "benchmark.json"
             --print-freq             "1"
             --training-only
             --data-backend "synthetic"
             --workers                "64"  
           )

PyTorch_maskrcnn_FP32_PARAMS=(
             "benchmark/Segmentation/MaskRCNN/pytorch"
             args
             --config-file            "/workspace/patch/e2e_mask_rcnn_R_50_FPN_1x.yaml"
             SOLVER.IMS_PER_BATCH     "$(scale 0.8)"
             DTYPE                    "float32"
             SOLVER.MAX_ITER          "400"
             OUTPUT_DIR               "/results"
             PATHS_CATALOG            "/workspace/patch/paths_catalog_ci.py"
           )

PyTorch_maskrcnn_FP16_PARAMS=(
             "benchmark/Segmentation/MaskRCNN/pytorch"
             args      
             --config-file            "/workspace/patch/e2e_mask_rcnn_R_50_FPN_1x.yaml"
             SOLVER.IMS_PER_BATCH     "$(scale 1.6)"
             DTYPE                    "float16"
             SOLVER.MAX_ITER          "400"
             OUTPUT_DIR               "/results"
             PATHS_CATALOG            "/workspace/patch/paths_catalog_ci.py"
           )


PyTorch_gnmt_FP32_PARAMS=(
            "benchmark/Translation/GNMT"
            args
            --dataset-dir             "/data/gnmt/wmt16_de_en"
            --train-batch-size        "$(scale 8 0.7)"
            --math                    "fp32"
            --epochs                  "1"
            --seed                    "2"
            --train-loader-workers    "64"
           )

PyTorch_gnmt_FP16_PARAMS=(
            "benchmark/Translation/GNMT"
            args
            --dataset-dir             "/data/gnmt/wmt16_de_en"
            --train-batch-size        "$(scale 16 0.7)"
            --math                    "fp16"
            --epochs                  "1"
            --seed                    "2"
            --train-loader-workers    "64"
           )

PyTorch_ncf_FP32_PARAMS=(
            "benchmark/Recommendation/NCF"
            args
            --data                    "/data/ncf/cache/ml-20m"
            --epochs                  "2"
            --batch_size              "$(scale 80000 0.7)"
           )

PyTorch_ncf_FP16_PARAMS=(
            "benchmark/Recommendation/NCF"
            args
            --data                    "/data/ncf/cache/ml-20m"
            --epochs                  "2"
            --batch_size              "$(scale 160000 0.7)"
            --amp
           )

PyTorch_transformerxlbase_FP32_PARAMS=(
            "benchmark/LanguageModeling/Transformer-XL/pytorch"
            args
            --data                    "/data/transformer-xl/wikitext-103"
            --max_step                "400"
            --batch_size              "$(scale 0.5 1.0)"
            --dataset                 "wt103" 
            --n_layer                 "16"
            --d_model                 "512"
            --n_head                  "8"
            --d_head                  "64"
            --d_inner                 "2048"
            --dropout                 "0.1"
            --dropatt                 "0.0"
            --optim                   "jitlamb"
            --lr                      "0.0"
            --eta_min                 "0.001"
            --warmup_step             "1000"
            --tgt_len                 "192"
            --mem_len                 "192"
            --eval_tgt_len            "192"
            --log_interval            "10"
            --eval_interval           "5000"
            --no_eval
            --roll
            --cuda
           )

PyTorch_transformerxlbase_FP16_PARAMS=(
            "benchmark/LanguageModeling/Transformer-XL/pytorch"
            args
            --data                    "/data/transformer-xl/wikitext-103"
            --max_step                "400"
            --batch_size              "$(scale 0.75 1.0)"
            --dataset                 "wt103" 
            --n_layer                 "16"
            --d_model                 "512"
            --n_head                  "8"
            --d_head                  "64"
            --d_inner                 "2048"
            --dropout                 "0.1"
            --dropatt                 "0.0"
            --optim                   "jitlamb"
            --lr                      "0.0"
            --eta_min                 "0.001"
            --warmup_step             "1000"
            --tgt_len                 "192"
            --mem_len                 "192"
            --eval_tgt_len            "192"
            --log_interval            "10"
            --eval_interval           "5000"
            --no_eval
            --roll
            --cuda
            --fp16
           )

PyTorch_transformerxllarge_FP32_PARAMS=(
            "benchmark/LanguageModeling/Transformer-XL/pytorch"
            args
            --data                    "/data/transformer-xl/wikitext-103"
            --max_step                "400"
            --batch_size              "$(scale 0.125 1.0)"
            --dataset                 "wt103" 
            --n_layer                 "18"
            --d_model                 "1024"
            --n_head                  "16"
            --d_head                  "64"
            --d_inner                 "4096"
            --dropout                 "0.2"
            --dropatt                 "0.2"
            --optim                   "adam"
            --lr                      "0.0"
            --warmup_step             "16000"
            --tgt_len                 "256"
            --mem_len                 "256"
            --eval_tgt_len            "128"
            --eval_interval           "5000"
            --no_eval
            --roll
            --cuda
           )

PyTorch_transformerxllarge_FP16_PARAMS=(
            "benchmark/LanguageModeling/Transformer-XL/pytorch"
            args
            --data                    "/data/transformer-xl/wikitext-103"
            --max_step                "400"
            --batch_size              "$(scale 0.25 1.0)"
            --dataset                 "wt103" 
            --n_layer                 "18"
            --d_model                 "1024"
            --n_head                  "16"
            --d_head                  "64"
            --d_inner                 "4096"
            --dropout                 "0.2"
            --dropatt                 "0.2"
            --optim                   "adam"
            --lr                      "0.0"
            --warmup_step             "16000"
            --tgt_len                 "256"
            --mem_len                 "256"
            --eval_tgt_len            "128"
            --eval_interval           "5000"
            --no_eval
            --cuda
            --fp16
           )

PyTorch_tacotron2_FP32_PARAMS=(
            "benchmark/SpeechSynthesis/Tacotron2"
            args
            -o                        "./"
            --model-name              "Tacotron2"
            --learning-rate           "0.0" 
            --epochs                  "2" 
            --batch-size              "$(scale 3 0.6)" 
            --weight-decay            "1e-6" 
            --grad-clip-thresh        "1.0"
            --log-file                "nvlog.json"
            --training-files          "filelists/ljs_audio_text_train_subset_625_filelist.txt"
            --dataset-path            "/data/tacotron2/LJSpeech-1.1"
            --cudnn-enabled
           )

PyTorch_tacotron2_FP16_PARAMS=(
            "benchmark/SpeechSynthesis/Tacotron2"
            args
            -o                        "./"
            --model-name              "Tacotron2"
            --learning-rate           "0.0" 
            --epochs                  "3" 
            --batch-size              "$(scale 3.24 0.6)"
            --weight-decay            "1e-6" 
            --grad-clip-thresh        "1.0"
            --log-file                "nvlog.json"
            --training-files          "filelists/ljs_audio_text_train_subset_625_filelist.txt"
            --dataset-path            "/data/tacotron2/LJSpeech-1.1"
            --cudnn-enabled
            --amp
           )


PyTorch_waveglow_FP32_PARAMS=(
            "benchmark/SpeechSynthesis/Tacotron2"
            args      
            -o                        "./"
            --model-name              "WaveGlow"
            --learning-rate           "0.0" 
            --epochs                  "2" 
            --segment-length          "8000"
            --batch-size              "$(scale 0.65 0.6)" 
            --weight-decay            "0" 
            --grad-clip-thresh        "65504"
            --log-file                "nvlog.json"
            --training-files          "filelists/ljs_audio_text_train_subset_625_filelist.txt"
            --dataset-path            "/data/tacotron2/LJSpeech-1.1"
            --cudnn-enabled
            --cudnn-benchmark
           )

PyTorch_waveglow_FP16_PARAMS=(
            "benchmark/SpeechSynthesis/Tacotron2"
            args      
            -o                        "./"
            --model-name              "WaveGlow"
            --learning-rate           "0.0" 
            --epochs                  "2" 
            --segment-length          "8000"
            --batch-size              "$(scale 0.9375 0.6)" 
            --weight-decay            "0" 
            --grad-clip-thresh        "65504"
            --log-file                "nvlog.json"
            --training-files          "filelists/ljs_audio_text_train_subset_1250_filelist.txt"
            --dataset-path            "/data/tacotron2/LJSpeech-1.1"
            --cudnn-enabled
            --cudnn-benchmark
            --amp
           )

PyTorch_bert_base_squad_FP32_PARAMS=(
            "benchmark/LanguageModeling/BERT"
            args
            "/data/bert_base/bert_base_uncased.pt"
            "2.0"
            "$(scale 1.5 0.7)"
            "0.0"
            "0.1"
            "fp32"
            "${NUM_GPU}"
            "1"
            "/data/squad/v1.1"
            "/data/bert_base/bert-base-uncased-vocab.txt"
            "."
            "train"
            "/data/bert_base/bert_config.json"
            "100"
)

PyTorch_bert_base_squad_FP16_PARAMS=(
            "benchmark/LanguageModeling/BERT"
            args      
            "/data/bert_base/bert_base_uncased.pt"
            "2.0"
            "$(scale 3 0.69)"
            "0.0"
            "0.1"
            "fp16"
            "${NUM_GPU}"
            "1"
            "/data/squad/v1.1"
            "/data/bert_base/bert-base-uncased-vocab.txt"
            "."
            "train"
            "/data/bert_base/bert_config.json"
            "100"
)

PyTorch_bert_large_squad_FP32_PARAMS=(
            "benchmark/LanguageModeling/BERT"
            args      
            "/data/bert_large/bert_large_uncased.pt"
            "2.0"
            "$(scale 0.5 0.7)"
            "0.0"
            "0.1"
            "fp32"
            "${NUM_GPU}"
            "1"
            "/data/squad/v1.1"
            "/data/bert_large/bert-large-uncased-vocab.txt"
            "."
            "train"
            "/data/bert_large/bert_config.json"
            "100"
)

PyTorch_bert_large_squad_FP16_PARAMS=(
            "benchmark/LanguageModeling/BERT"
            args      
            "/data/bert_large/bert_large_uncased.pt"
            "2.0"
            "$(scale 1 0.7)"
            "0.0"
            "0.1"
            "fp16"
            "${NUM_GPU}"
            "1"
            "/data/squad/v1.1"
            "/data/bert_large/bert-large-uncased-vocab.txt"
            "."
            "train"
            "/data/bert_large/bert_config.json"
            "100"
)
