# Employee Attrition Early-Warning Model

An early-warning model that flags employees who may be at risk of leaving, so managers can check in sooner.

This is a Columbia AI bootcamp project (Module 19). It trains one neural network with two outputs: **attrition** (will this employee leave?) and **department fit** (which of three departments does this employee's profile most resemble?). It is a learning project, not a production tool, and the results below show why.

## Repository layout

```
.
├── notebooks/
│   └── employee_attrition_model.ipynb   # data prep, model, training, evaluation
├── requirements.txt
└── README.md
```

## Data

- 1,470 employee records with 27 columns, loaded straight from the bootcamp's URL in the first cell (no local copy needed). It appears to be a version of IBM's widely used synthetic HR attrition dataset, so the people in it are not real employees.
- The model uses 11 numeric inputs: age, distance from home, education, environment satisfaction, job involvement, job level, job satisfaction, number of companies worked for, percent salary hike, total working years and work-life balance.
- The labels are imbalanced. **16.1%** of employees left. Departments: Research & Development 65.4%, Sales 30.3%, Human Resources 4.3%.

## Model

- Two shared dense layers (64 → 128, ReLU), then one branch per task (32 units, ReLU) ending in an output layer.
- Adam optimizer, 100 epochs. Losses: categorical cross-entropy for department, binary cross-entropy for attrition.
- 75/25 train/test split (368 test employees). **No random seed is set**, so every run gives slightly different numbers.

## Results (October 2026 re-run)

These numbers come from re-running the notebook top to bottom. To see how much they move between runs, I also ran the same pipeline five more times outside the notebook. The "baseline" column is what you would get by always guessing the most common answer.

| Task | Notebook run | 5 extra runs (range) | Always guess the most common answer |
|---|---|---|---|
| Attrition accuracy | 79.9% | 76.9% – 80.2% | 79.3% in the notebook run (79.6% – 84.0% across the extra runs) |
| Leavers actually caught (recall on "Yes") | 22 of 76 (29%) | 16% – 29% | 0% |
| Department-fit accuracy | 60.1% | 54.9% – 64.4% | 65.8% in the notebook run (63.6% – 68.2% across the extra runs) |

What the notebook run shows in more detail:

- **Attrition:** of the 76 test employees who actually left, the model flagged 22 and missed 54. Of the 42 employees it flagged, 22 (52%) actually left.
- **Department fit:** recall by department was R&D 76%, Sales 34% and Human Resources 6% (1 of 18).

### How to read this

**Attrition.** The 80% accuracy sounds good, but about 80% of employees stay, so a rule that always says "No one is leaving" scores about the same. Accuracy hides the part that matters: the model misses roughly 7 out of 10 people who actually leave. As an early-warning signal it is weak. It finds some at-risk employees, but it should not be used to reassure anyone that an employee *isn't* at risk.

**Department fit is not reliable.** Accuracy was 60.1% in the notebook run and 55% – 64% across the repeat runs. That is **worse than always guessing "Research & Development"** (about 66%). In practice it recognises R&D profiles, sometimes recognises Sales and almost never recognises Human Resources. Likely reasons:

- None of the 11 inputs describe what someone does or is good at (role, skills, education field). They describe tenure, satisfaction and commute, which say little about department.
- Human Resources has only about 63 employees in the whole dataset, too few to learn from.
- The model overfits. Training accuracy reached nearly 100% while test accuracy stayed near 60%.
- The output layer uses `sigmoid` with categorical cross-entropy. For choosing one of three departments, `softmax` is the standard choice. I have left this unchanged so the results match the original project, and it is noted under "Possible improvements".

I would not use the department-fit output for anything beyond showing the technique.

### Differences from the original notebook

- The original notebook reported **Attrition 80.2%** and **Department 59.8%**. The re-run gives 79.9% and 60.1%, within normal run-to-run variation because there is no random seed.
- With current Keras (3.15), `model.evaluate` returns its results in a different order. The original print cell then labelled two **loss** values as accuracies ("Department Accuracy: 1.79"). I fixed it to look metrics up by name. The model and the data are unchanged.
- I added a "Sanity check" section at the end of the notebook with the baselines, confusion matrix and per-class results above. It does not change the model.
- The original README answer said sigmoid is "appropriate for multi-class" classification. For picking one department out of three, softmax is the better fit, so that answer has been corrected below.

## What this means for a client

**The AI points; a manager decides.**

At best, a model like this is a prompt to start a conversation. It is never a verdict. The right order is:

1. **First Brain: human judgment.** The manager knows the person, the context and what is going on in their life and work. That comes first.
2. **Third Brain: AI as a second look.** The model can suggest "you may want to check in with this person sooner." The manager decides whether that makes sense and what, if anything, to do.

In its current state this model misses most people who leave. A client should treat a "not flagged" result as meaningless, and treat a "flagged" result as a reason to listen, not to act.

## Limitations

- Small, imbalanced, synthetic dataset. The results do not carry over to any real company without retraining and validating on that company's own data.
- Accuracy is the wrong headline metric for both tasks because of class imbalance. Recall on leavers and comparison with a baseline matter more.
- No random seed, no validation split, no early stopping and no hyperparameter tuning, so the results are noisy and the model overfits.
- Only 11 of the 26 available features are used. Features known to relate to attrition in this dataset, such as overtime, job role and marital status, were left out because they are not numeric.
- The model gives no explanation for an individual prediction.

## Ethics note

**This kind of model should never be the sole basis for an HR decision.** That covers hiring, firing, promotion, pay, reassignment and how someone is treated.

- Being "flagged" is a statistical guess, not a fact about a person. Acting on it alone risks self-fulfilling outcomes, such as an employee being passed over because a model said they might leave.
- Inputs such as age and marital status can stand in for protected characteristics. Any real use would need a fairness review, legal and HR sign-off, and clear communication to employees about what data is used and why.
- People affected by a decision should be able to understand it and challenge it. This model cannot explain itself.
- Use it to start supportive check-ins, not to rank, monitor or penalise people.

## Possible improvements

- Use `softmax` for the department output, and set a random seed so results can be reproduced.
- Add a validation split with early stopping and dropout to reduce overfitting.
- Handle class imbalance with class weights or resampling, and tune the attrition decision threshold to catch more leavers.
- Encode and include categorical features (overtime, job role, business travel, marital status), with a fairness check on sensitive ones.
- Compare against a simple logistic regression. On data this small it may do as well as the neural network or better, and it is easier to explain.

## Summary questions (from the original assignment)

1. **Is accuracy the best metric?** No. About 84% of employees stay and about 65% are in R&D, so a model can reach high accuracy just by guessing the majority answer. Recall and precision on the "Yes" attrition class, per-class results and a comparison with a baseline give a truer picture.
2. **Which output activations?** The project used `sigmoid` for both outputs. That works for the two-class attrition output. For department, where an employee belongs to exactly one of three classes, `softmax` would be more appropriate.
3. **How could the model be improved?** See "Possible improvements" above.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/employee_attrition_model.ipynb
```

The notebook downloads its data from `static.bc-edx.com`, so it needs internet access. Expect your numbers to differ slightly from the ones above.

## Sources

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [TensorFlow / Keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
