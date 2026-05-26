# Column Split Data Understanding Notes

## Scope

- Original CSV is not modified.
- Modeling/split datasets exclude `PID` and `KA_name`.
- Each original feature is exported as a `feature + CHURN` CSV.
- Category values such as `Bronze` and `SOHO` are exported as separate churn datasets.

## Modeling Interpretation

Single-column screening is a quick limit check, not the final model. If one feature alone cannot get high F1/PR-AUC, the final model needs interaction features, leakage-safe target encoding, and imbalance-aware threshold tuning.

## Best Single-Column Signals

| column | best model | F1 | recall | precision | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Total_SUBs | DecisionTree_balanced | 0.1498 | 0.3909 | 0.0927 | 0.0781 |
| AvgMobileRevenue | LogisticRegression_balanced | 0.1441 | 0.4545 | 0.0856 | 0.0837 |
| Active_subscribers | LogisticRegression_balanced | 0.1426 | 0.4000 | 0.0868 | 0.0829 |
| TotalRevenue | LogisticRegression_balanced | 0.1416 | 0.4455 | 0.0842 | 0.0843 |
| EffectiveSegment | DecisionTree_balanced | 0.1346 | 0.3273 | 0.0847 | 0.0714 |
| CRM_PID_Value_Segment | LogisticRegression_balanced | 0.1276 | 0.5818 | 0.0717 | 0.0696 |
| Billing_ZIP | DecisionTree_balanced | 0.1276 | 0.7000 | 0.0702 | 0.0822 |
| Not_Active_subscribers | DecisionTree_balanced | 0.1259 | 0.2455 | 0.0846 | 0.0727 |

## Next Paper Comparison Checkpoints

- Compare paper preprocessing against this split-first setup.
- Check whether paper used `PID` or `KA_name`; this workflow excludes both from model datasets.
- Verify how the paper handled `Billing_ZIP`, missing values, skewed subscriber counts, and class imbalance.
- For target/risk encoding, compute rates inside train folds only to avoid target leakage.
