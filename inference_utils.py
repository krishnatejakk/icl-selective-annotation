import json
import os


def inference_for_xsum(
    tokenizer_gpt,
    inference_model,
    device,
    prompt_cache_dir,
    output_dir,
    file,
):
    with open(os.path.join(prompt_cache_dir, file)) as f:
        one_test_example = json.load(f)

    context = one_test_example[1]
    input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
    input_ids = input_ids[:, :1900]
    input_len = input_ids.shape[1]
    input_ids = input_ids.to(device)
    gen_tokens = inference_model.generate(
        input_ids,
        do_sample=False,
        temperature=0.7,
        max_length=input_len + 64,
        output_scores=True,
        return_dict_in_generate=True,
    )
    generated_text = tokenizer_gpt.batch_decode(gen_tokens.sequences.view(-1, 1))
    stop = ["--", "\n", ";", "#"]
    stop_index = len(generated_text)
    for i, c in enumerate(generated_text):
        if i > input_len and c.strip(" ") in stop:
            stop_index = i
            break
    prediction = " ".join(generated_text[input_len:stop_index])
    gold = one_test_example[2]["summary"]
    pred = prediction
    with open(f"{output_dir}/{file}", "w") as f:
        json.dump(
            [
                " ".join(generated_text[input_len:]),
                " ".join(generated_text[input_len:stop_index]),
                gold,
                input_len,
                stop_index,
            ],
            f,
            indent=4,
        )

    return gold, pred
