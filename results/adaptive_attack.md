# Adaptive black-box attacker (greedy, queries the model, may change up to 30% of words)

Sample: 300 test spam messages each model initially detects.

| Model | Evasion success | Words changed (successful) | Most-used tricks |
|---|---|---|---|
| B0_paper_style_svm | 0.640 | 4.16 | charswap (570), typo (210), homoglyph (17) |
| A1_hybrid_no_norm | 0.213 | 5.08 | spacing (182), fullwidth (82), typo (24) |
| A4_no_norm_advtrain | 0.053 | 3.81 | typo (21), charswap (11), leet (9) |
| A5_no_spellfix_advtrain | 0.030 | 2.89 | typo (17), charswap (7), leet (2) |
| SMSGuard | 0.010 | 1.67 | typo (3), charswap (2) |
