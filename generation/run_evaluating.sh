user_result_file_path=""



export CUDA_VISIBLE_DEVICES=1
python evaluating.py --user_result_file_path=$user_result_file_path \
--golden_response_file_path=$golden_response_file_path \
