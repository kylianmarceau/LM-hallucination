import numpy as np

student_number = "DBNKYL001"

seed = int.from_bytes(
    student_number.encode("utf-8"),
    byteorder="big"
)

rng = np.random.default_rng(seed=seed)

print(f"Student number: {student_number}")
print(f"Random seed: {seed}")