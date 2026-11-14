# E4: generalisation to attacks never seen in adversarial training

Each row trains with every attack except the one named (or only the base paper's three attacks), then measures detection of the held-out attack. `no_norm` = same model without normalisation.

| Held-out attack | SMSGuard | no_norm + adv. training |
|---|---|---|
| spacing | 0.982 | 0.973 |
| homoglyph | 0.994 | 0.985 |
| charswap | 0.985 | 0.987 |
| typo | 0.985 | 0.986 |
| zero_width | 0.990 | 0.982 |
| leet | 0.984 | 0.986 |
| fullwidth | 0.987 | 0.975 |
| emoji_wedge | 0.975 | 0.981 |

## Trained only on the base paper's attacks (spacing, homoglyph, charswap), tested on new ones

| Attack | SMSGuard | no_norm + adv. training |
|---|---|---|
| typo | 0.986 | 0.986 |
| zero_width | 0.990 | 0.978 |
| leet | 0.985 | 0.984 |
| fullwidth | 0.988 | 0.971 |
| emoji_wedge | 0.986 | 0.979 |
| hybrid | 0.993 | 0.986 |
