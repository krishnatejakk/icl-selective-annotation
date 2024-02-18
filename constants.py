TASK_NAMES = [
    "mnli",
    "rte",
    "sst5",
    "mrpc",
    "dbpedia_14",
    "hellaswag",
    "xsum",
    # "nq", # Needs OpenAI API
]


QUERYLESS_SUBMODLIB_FUNCTIONS = [
    # "ConcaveOverModularFunction",
    "DisparityMinFunction",
    "DisparitySumFunction",
    # "FacilityLocationConditionalGainFunction",
    # "FacilityLocationConditionalMutualInformationFunction",
    "FacilityLocationFunction",
    # "FacilityLocationMutualInformationFunction",
    # "FacilityLocationVariantMutualInformationFunction",
    # "FeatureBasedFunction",
    # "GraphCutConditionalGainFunction",
    "GraphCutFunction",
    # "GraphCutMutualInformationFunction",
    # "LogDeterminantConditionalGainFunction",
    # "LogDeterminantConditionalMutualInformationFunction",
    "LogDeterminantFunction",
    # "LogDeterminantMutualInformationFunction",
    # "ProbabilisticSetCoverConditionalGainFunction",
    # "ProbabilisticSetCoverConditionalMutualInformationFunction",
    # "ProbabilisticSetCoverFunction",
    # "ProbabilisticSetCoverMutualInformationFunction",
    # "SetCoverConditionalGainFunction",
    # "SetCoverConditionalMutualInformationFunction",
    # "SetCoverFunction",
    # "SetCoverMutualInformationFunction",
]


def camel_to_snake(strings):
    converted_strings = []
    for string in strings:
        converted_string = ""
        for index, char in enumerate(string):
            if char.isupper() and index != 0:
                converted_string += "_" + char.lower()
            else:
                converted_string += char.lower()
        converted_strings.append(converted_string)
    return converted_strings


SELECTIVE_ANNOTATION_METHODS = [
    "random",
    "diversity",
    "fast_votek",
    "mfl",
    "votek",
    "least_confidence",
    *camel_to_snake(QUERYLESS_SUBMODLIB_FUNCTIONS),
]
