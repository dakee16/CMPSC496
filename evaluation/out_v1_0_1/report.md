# Answer-checking benchmark (schema 1.0.1)

- cases: 503 (503 scored, 0 ambiguous)
- accuracy: **0.690**
- false acceptance: 0.0000
- false rejection: 0.0000
- indeterminate: 0.3101
- attempts wrongly consumed: 0

## Tier distribution

- `execution-final`: 159
- `execution-reference`: 32
- `policy`: 52
- `syntax`: 104
- `system`: 156

## Accuracy by category

- `blank`: 1.000 (n=52)
- `boundary_off_by_one`: 0.342 (n=41)
- `clean_logical_error`: 0.281 (n=32)
- `exact_reference`: 1.000 (n=52)
- `no_op`: 0.423 (n=52)
- `policy_violation`: 1.000 (n=52)
- `renamed_variables`: 1.000 (n=9)
- `runtime_error`: 0.423 (n=52)
- `syntax_error`: 1.000 (n=52)
- `timeout`: 0.423 (n=52)
- `wrong_after_divergent_prefix`: 0.704 (n=27)
- `wrong_after_revealed_prefix`: 0.733 (n=30)
