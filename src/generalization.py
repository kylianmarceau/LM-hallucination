QUESTION_TYPES = {"present", "absent_random", "absent_adversarial"}

def cross_category_split(results, train_types, test_types, val_indices):
    # keep training outside the saved validation set
    train_types = set(train_types)
    test_types = set(test_types)

    if not train_types or not test_types:
        raise ValueError("training and test question types must not be empty")
    if (train_types | test_types) - QUESTION_TYPES:
        raise ValueError("unknown training or test question type")

    for index in val_indices:
        if type(index) is not int or not 0 <= index < len(results):
            raise ValueError("validation indices must be valid result indices")

    held_out = set(val_indices)
    if len(held_out) != len(val_indices):
        raise ValueError("validation indices must not contain duplicates")

    train_indices = []
    test_indices = []

    for index, result in enumerate(results):
        if result.parsed_answer is None:
            continue

        if index not in held_out and result.question_type in train_types:
            train_indices.append(index)
        if index in held_out and result.question_type in test_types:
            test_indices.append(index)

    if not train_indices or not test_indices:
        raise ValueError("the selected question types produce an empty train or test set")

    return train_indices, test_indices
