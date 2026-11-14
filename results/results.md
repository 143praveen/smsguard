# E1/E2 results (seed 42, train 53530, test 13383, test spam 5219)

## Clean test set

| Model | Acc | Prec | Recall | F1 | FPR | ROC-AUC | FPR (ham+typos) | Train s |
|---|---|---|---|---|---|---|---|---|
| B0_paper_style_svm | 0.9796 | 0.9806 | 0.9669 | 0.9737 | 0.0122 | 0.9958 | 0.0120 | 0.7 |
| A1_hybrid_no_norm | 0.9928 | 0.9917 | 0.9897 | 0.9907 | 0.0053 | 0.9994 | 0.0049 | 13.9 |
| A2_hybrid_norm | 0.9922 | 0.9917 | 0.9883 | 0.9900 | 0.0053 | 0.9994 | 0.0051 | 37.7 |
| A3_hybrid_norm_meta | 0.9928 | 0.9921 | 0.9893 | 0.9907 | 0.0050 | 0.9995 | 0.0055 | 55.6 |
| A4_no_norm_advtrain | 0.9927 | 0.9921 | 0.9891 | 0.9906 | 0.0050 | 0.9992 | 0.0080 | 22.5 |
| A5_no_spellfix_advtrain | 0.9930 | 0.9946 | 0.9874 | 0.9910 | 0.0034 | 0.9995 | 0.0059 | 75.2 |
| A6_no_urlrep_advtrain | 0.9928 | 0.9944 | 0.9872 | 0.9908 | 0.0036 | 0.9994 | 0.0047 | 81.2 |
| SMSGuard | 0.9933 | 0.9952 | 0.9875 | 0.9913 | 0.0031 | 0.9995 | 0.0047 | 84.7 |

## Spam detection rate under attack (30% of words perturbed)

Attacks after `charswap` were not evaluated by the base paper.

| Model | none | spacing | homoglyph | charswap | typo | zero_width | leet | fullwidth | emoji_wedge | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_paper_style_svm | 0.967 | 0.869 | 0.874 | 0.875 | 0.858 | 0.869 | 0.910 | 0.870 | 0.869 | 0.876 |
| A1_hybrid_no_norm | 0.990 | 0.967 | 0.984 | 0.985 | 0.985 | 0.970 | 0.984 | 0.965 | 0.970 | 0.977 |
| A2_hybrid_norm | 0.988 | 0.988 | 0.988 | 0.988 | 0.986 | 0.988 | 0.989 | 0.988 | 0.988 | 0.989 |
| A3_hybrid_norm_meta | 0.989 | 0.985 | 0.995 | 0.988 | 0.987 | 0.992 | 0.987 | 0.989 | 0.991 | 0.991 |
| A4_no_norm_advtrain | 0.989 | 0.995 | 0.990 | 0.990 | 0.987 | 0.998 | 0.992 | 0.993 | 0.995 | 0.994 |
| A5_no_spellfix_advtrain | 0.987 | 0.997 | 0.995 | 0.988 | 0.985 | 1.000 | 0.991 | 1.000 | 0.999 | 0.998 |
| A6_no_urlrep_advtrain | 0.987 | 0.997 | 0.995 | 0.987 | 0.986 | 1.000 | 0.992 | 1.000 | 0.998 | 0.998 |
| SMSGuard | 0.988 | 0.997 | 0.995 | 0.988 | 0.986 | 1.000 | 0.991 | 1.000 | 0.999 | 0.998 |

## McNemar test vs SMSGuard (clean test set)

| Model | SMSGuard right, model wrong | model right, SMSGuard wrong | p-value |
|---|---|---|---|
| B0_paper_style_svm | 218 | 35 | 1.74e-33 |
| A1_hybrid_no_norm | 28 | 21 | 3.92e-01 |
| A2_hybrid_norm | 33 | 19 | 7.04e-02 |
| A3_hybrid_norm_meta | 18 | 11 | 2.65e-01 |
| A4_no_norm_advtrain | 35 | 27 | 3.74e-01 |
| A5_no_spellfix_advtrain | 6 | 2 | 2.89e-01 |
| A6_no_urlrep_advtrain | 6 | 0 | 3.12e-02 |
