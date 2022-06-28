import numpy as np
from EncoderDecoderDataloaders import get_shared_examples

from ArkDataset.load_ark import load_ark
from TweeBankDataset.load_tweebank import load_tweebank


def expert_baseline_numbers():
    ark_train, ark_val, ark_test = load_ark()
    tweebank_train, tweebank_val, tweebank_test = load_tweebank()
    ark_all = ark_train + ark_val + ark_test

    tweebank_all = tweebank_train + tweebank_val + tweebank_test

    shared_examples = get_shared_examples(ark_all, tweebank_all)

    ark_to_tweebank_mapping = dict()
    tweebank_to_ark_mapping = dict()
    error_count_ark_to_tweebank = 0
    error_count_tweebank_to_ark = 0
    total_count = 0

    ark_set = set()
    twee_set = set()

    for a, b, c in shared_examples:
        b = np.array(b)
        c = np.array(c)

        for i, j in zip(b, c):
            ark_set.add(i)
            twee_set.add(j)
            total_count += 1

            if i not in ark_to_tweebank_mapping:
                ark_to_tweebank_mapping[i] = j
            else:
                if ark_to_tweebank_mapping[i] != j:
                    error_count_ark_to_tweebank += 1

            if j not in tweebank_to_ark_mapping:
                tweebank_to_ark_mapping[j] = i
            else:
                if tweebank_to_ark_mapping[j] != i:
                    error_count_tweebank_to_ark += 1

    ark_to_tweebank_mapping_maximal = dict()
    tweebank_to_ark_mapping_maximal = dict()
    error_count_ark_to_tweebank_maximal = 0
    error_count_tweebank_to_ark_maximal = 0
    total_count_maximal = 0

    for a, b, c in shared_examples:
        b = np.array(b)
        c = np.array(c)

        for i, j in zip(b, c):

            if i not in ark_to_tweebank_mapping_maximal:
                ark_to_tweebank_mapping_maximal[i] = dict()

            if j not in ark_to_tweebank_mapping_maximal[i]:
                ark_to_tweebank_mapping_maximal[i][j] = 1
            else:
                ark_to_tweebank_mapping_maximal[i][j] += 1

            if j not in tweebank_to_ark_mapping_maximal:
                tweebank_to_ark_mapping_maximal[j] = dict()

            if i not in tweebank_to_ark_mapping_maximal[j]:
                tweebank_to_ark_mapping_maximal[j][i] = 1
            else:
                tweebank_to_ark_mapping_maximal[j][i] += 1

    for k, v in ark_to_tweebank_mapping_maximal.items():
        max_v_prime = 0
        max_k_prime = 0
        for k_prime, v_prime in v.items():
            if v_prime >= max_v_prime:
                max_v_prime = v_prime
                max_k_prime = k_prime

        ark_to_tweebank_mapping_maximal[k] = max_k_prime

    for k, v in tweebank_to_ark_mapping_maximal.items():
        max_v_prime = 0
        max_k_prime = 0
        for k_prime, v_prime in v.items():
            if v_prime >= max_v_prime:
                max_v_prime = v_prime
                max_k_prime = k_prime

        tweebank_to_ark_mapping_maximal[k] = max_k_prime

    for a, b, c in shared_examples:
        b = np.array(b)
        c = np.array(c)

        for i, j in zip(b, c):

            total_count_maximal += 1

            if i not in ark_to_tweebank_mapping_maximal:
                ark_to_tweebank_mapping_maximal[i] = j
            else:
                if ark_to_tweebank_mapping_maximal[i] != j:
                    error_count_ark_to_tweebank_maximal += 1

            if j not in tweebank_to_ark_mapping_maximal:
                tweebank_to_ark_mapping_maximal[j] = i
            else:
                if tweebank_to_ark_mapping_maximal[j] != i:
                    error_count_tweebank_to_ark_maximal += 1

    # print(f'Accuracy ark to tweebank original = {100*(1 - error_count_ark_to_tweebank/total_count):.2f}%')
    print(
        f"Accuracy ark to tweebank maximal = {100*(1 - error_count_ark_to_tweebank_maximal/total_count_maximal):.2f}%"
    )
    # print(f'Accuracy tweebank to ark original = {100*(1 - error_count_tweebank_to_ark/total_count):.2f}%')
    print(
        f"Accuracy tweebank to ark maximal = {100*(1 - error_count_tweebank_to_ark_maximal/total_count_maximal):.2f}%"
    )


# Accuracy ark to tweebank maximal = 83.81%
# Accuracy tweebank to ark maximal = 88.97%
if __name__ == "__main__":
    expert_baseline_numbers()
