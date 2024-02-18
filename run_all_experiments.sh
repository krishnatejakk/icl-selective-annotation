#!/bin/bash

python main.py --task_name mnli --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name rte --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name sst5 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name mrpc --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name dbpedia_14 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name hellaswag --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name xsum --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/log_determinant_function --model_name=EleutherAI/gpt-j-6B

python main.py --task_name nq --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method disparity_min_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/disparity_min_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method disparity_sum_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/disparity_sum_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method facility_location_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/facility_location_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method graph_cut_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/graph_cut_function --model_name=EleutherAI/gpt-j-6B
python main.py --task_name nq --selective_annotation_method log_determinant_function --model_cache_dir models --data_cache_dir datasets --output_dir outputs/nq/log_determinant_function --model_name=EleutherAI/gpt-j-6B


python exelify_results.py
