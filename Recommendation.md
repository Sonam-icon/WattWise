# Recommendation — practical energy insight for customers and utilities

## Opportunity

A utility could use appliance-load estimates to provide customers with broad insights into flexible household electricity use. Washing-machine operation is often shiftable within a day, so an opt-in programme could encourage customers to move suitable loads away from locally defined peak-demand periods, where tariff design and customer needs make that useful.

The REFIT analysis suggests that washing-machine usage is intermittent and that usage patterns differ across households. This supports customer-level engagement and pilot testing, but not precise automated control based solely on aggregate-meter estimates.

## Recommended customer-facing intervention

Offer an opt-in notification or weekly report that:

- Summarises likely washing-machine usage windows rather than claiming exact appliance readings.
- Shows broad opportunities to shift a flexible cycle to a lower-demand or lower-cost period, if such a period exists under the customer's tariff.
- Lets the customer choose whether and when to participate; do not automatically delay or interrupt an appliance.
- Clearly labels estimates and communicates uncertainty.

## Utility pilot and impact measurement

Run a controlled pilot with a baseline group and an intervention group. Define the intervention period and primary outcome before launch. Compare:

1. Change in estimated washing-machine energy used during the defined peak window.
2. Change in household peak-period aggregate demand.
3. Customer participation, opt-out and engagement rates.
4. Customer-reported convenience and any complaints about suggested timing.

Where possible, validate appliance estimates against temporary appliance sub-meters or another reliable ground-truth source. Report estimation error and missing-data coverage alongside any claimed savings. Do not attribute a reduction in aggregate peak demand to washing-machine shifting unless the evaluation design supports that conclusion.

## Appropriate uses and safeguards

**Potentially appropriate:** aggregate usage-pattern insights, identifying candidate households for an energy-advice pilot, and evaluating flexible-load engagement at group level.

**Not appropriate without further validation:** appliance-level billing, enforcement, automatic appliance control, or definitive claims about an individual customer's washing-machine operation. Appliance signatures overlap, behaviour varies, and the tested model has only moderate minute-level activity detection (test F1 approximately 0.45) and was evaluated on one household.

Use customer consent, data minimisation, clear explanations and an easy opt-out. Avoid exposing detailed household routines to parties that do not need them.

## Limitations and transferability

- REFIT represents UK homes and should not be assumed representative of Indian households, tariffs, appliance models or usage habits.
- The model was trained and evaluated on House 5; cross-house generalisation has not been demonstrated. Its two-stage approach reduced the total-energy error from 112.10 kWh down to 30.31 kWh (a 73.0% error reduction compared to the always-off baseline), but test activity detection remained moderate (precision 0.566, recall 0.376 and F1 0.452).
- The washing-machine channel has missing observations, and aggregate readings contain overlapping appliance loads.
- The 5 W activity threshold and event-merging rule are practical analysis choices, not universal definitions of a wash cycle.
- Estimated energy is not equivalent to a direct appliance sub-meter measurement.

## If only 15- or 30-minute smart-meter readings were available

At 15- or 30-minute resolution, short washing-machine phases and transitions would be averaged together. This would make it harder to identify cycle boundaries, short operating events, peak power and per-cycle energy. A coarser system would likely need to rely more heavily on daily/weekly usage patterns and household-level load profiles. Event-level disaggregation would be less reliable, so recommendations should be broader and uncertainty should be communicated more prominently. A model designed for one-minute data should not simply be applied to lower-frequency readings without retraining and re-evaluation.
