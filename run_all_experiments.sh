#!/bin/bash

python main.py --task_name mnli --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mnli --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mnli/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name rte --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name rte --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/rte/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name sst5 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name sst5 --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/sst5/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name mrpc --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name mrpc --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/mrpc/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name dbpedia_14 --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name dbpedia_14 --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/dbpedia_14/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name hellaswag --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name hellaswag --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/hellaswag/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B

python main.py --task_name xsum --selective_annotation_method random --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/random --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method diversity --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/diversity --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method fast_votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/fast_votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method mfl --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/mfl --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method votek --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/votek --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method least_confidence --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/least_confidence --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method DisparityMinFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/DisparityMinFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method DisparitySumFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/DisparitySumFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/FacilityLocationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method GraphCutFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/GraphCutFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method LogDeterminantFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/LogDeterminantFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/FacilityLocationMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method FacilityLocationVariantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/FacilityLocationVariantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method GraphCutMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/GraphCutMutualInformationFunction --model_name=EleutherAI/gpt-j-6B
python main.py --task_name xsum --selective_annotation_method LogDeterminantMutualInformationFunction --model_cache_dir models --data_cache_dir datasets --output_dir outputs/xsum/LogDeterminantMutualInformationFunction --model_name=EleutherAI/gpt-j-6B


python exelify_results.py
