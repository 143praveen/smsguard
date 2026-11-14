# False-positive stress test (share of legitimate messages wrongly flagged as spam)

`curated_link_ham` was written alongside the URL-reputation feature, so treat it as optimistic.

| Model | curated_hard_ham (n=40) | curated_link_ham (n=10) | test_ham | test_ham with link (n=32) | test_ham + emoji | test_ham + non-Latin name | spam-with-link recall (n=1619) |
|---|---|---|---|---|---|---|---|
| B0_paper_style_svm | 0.200 | 0.300 | 0.012 | 0.062 | 0.012 | 0.012 | 0.969 |
| A1_hybrid_no_norm | 0.125 | 0.200 | 0.005 | 0.062 | 0.005 | 0.005 | 0.999 |
| A4_no_norm_advtrain | 0.175 | 0.400 | 0.005 | 0.062 | 0.005 | 0.005 | 0.998 |
| A3_hybrid_norm_meta | 0.125 | 0.000 | 0.005 | 0.188 | 0.005 | 0.004 | 1.000 |
| A5_no_spellfix_advtrain | 0.075 | 0.000 | 0.003 | 0.125 | 0.003 | 0.003 | 1.000 |
| A6_no_urlrep_advtrain | 0.150 | 0.500 | 0.004 | 0.219 | 0.003 | 0.003 | 1.000 |
| SMSGuard | 0.100 | 0.000 | 0.003 | 0.125 | 0.003 | 0.002 | 1.000 |

## Curated messages wrongly flagged by SMSGuard

- Your appointment with Dr Patel is confirmed for Tue 14 May at 10:30am. Reply C to cancel.
- Your parcel has been delivered to your front door. Thanks for shopping with us.
- Reminder: your electricity bill of $84.20 is due on 28 Feb. Pay in the app as usual.
- Call me on 0412 345 678 when you're free
