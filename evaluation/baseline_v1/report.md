# Answer-checking benchmark (schema 1.0.0)

- cases: 535 (535 scored, 0 ambiguous)
- accuracy: **0.643**
- false acceptance: 0.0000
- false rejection: 0.0112
- indeterminate: 0.3458
- attempts wrongly consumed: 0

## Tier distribution

- `execution-final`: 162
- `execution-reference`: 32
- `policy`: 52
- `syntax`: 104
- `system`: 185

## Accuracy by category

- `blank`: 1.000 (n=52)
- `boundary_off_by_one`: 0.342 (n=41)
- `clean_logical_error`: 0.281 (n=32)
- `exact_reference`: 0.962 (n=52)
- `no_op`: 0.423 (n=52)
- `policy_violation`: 1.000 (n=52)
- `renamed_variables`: 0.132 (n=38)
- `runtime_error`: 0.423 (n=52)
- `syntax_error`: 1.000 (n=52)
- `timeout`: 0.423 (n=52)
- `wrong_after_divergent_prefix`: 0.733 (n=30)
- `wrong_after_revealed_prefix`: 0.733 (n=30)
