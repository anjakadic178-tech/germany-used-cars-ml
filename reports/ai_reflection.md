# D3 — AI Workflow Reflection

**Project:** Germany Used Cars 2023 — End-to-End Machine Learning Analysis  
**Author:** Anja Kadic  
**Date:** July 2026

---

## Overview

This document reflects honestly on how I used AI assistance throughout this machine
learning project — what I used it for, where it helped, where it had limitations,
and what I learned from the experience of working alongside an AI tool.

The AI tool I used was **Claude Code** (Anthropic), an AI assistant integrated
into the terminal that can read files, write code, and run commands.

---

## How I Used AI — Task by Task

### Data Cleaning

I used AI to help structure the twelve sequential cleaning steps in `src/clean.py`.
The key decision — that the filters had to be applied **one at a time** and not all
at once — came from understanding what "sequential" means for row-count accounting.
When an early version of the code applied all masks simultaneously, the row counts
were wrong (double-counted). I asked the AI to explain why, and once I understood
the problem I was able to verify the corrected version myself.

The decision about **which cars to remove** (pre-2000 vehicles, prices above €80,000,
zero-mileage listings) was guided by teacher instructions to build a general-market
model. The AI helped me implement those decisions in code, but the reasoning behind
the scope — why an old Porsche 911 behaves differently from a 2018 Golf — was
something I had to understand and explain myself.

### Feature Engineering

The idea to replace `year` with `car_age = 2023 − year` and to create
`mileage_per_year` came from the project guidelines. AI helped me implement them
correctly and explained why keeping both `year` and `car_age` in the model would
cause perfect collinearity. This was a concept I had to genuinely understand because
it relates to a core principle of linear models taught in the course.

### Model Training and Evaluation

I used AI to write the model ladder in `src/train.py` (Dummy → Logistic/Ridge →
Decision Tree → XGBoost). AI suggested using `RandomizedSearchCV` instead of
`GridSearchCV` for XGBoost tuning to reduce training time, which was practical
advice I verified by comparing the two approaches.

The most important thing I had to understand myself was **why F1_macro is the right
metric and F1_binary is misleading**. The Dummy classifier achieves F1_binary = 0.667
by always predicting HIGH — this looks good but is completely uninformative.
F1_macro = 0.333 is the honest number. I had to be able to explain this distinction
clearly, and the AI helped me check my understanding by asking it follow-up questions.

### Leakage Prevention

Preventing data leakage was the most important correctness requirement in the project.
AI helped me structure the pipeline so that the OneHotEncoder, StandardScaler, and
`price_segment` threshold are all fitted on training data only. I tested my
understanding by asking: "What would happen if we computed the threshold on the full
dataset?" — and the AI explained the consequence clearly enough that I could then
explain it in my own words.

### Streamlit App

AI wrote most of the initial Streamlit app code based on my instructions about what
the interface should look like and what features it should have. I reviewed every
section, tested it locally, and gave feedback when something did not work the way
I expected. For example, the initial mileage label said "Mileage (km)" which implied
exact mileage — I changed the requirement to "Maximum mileage" and directed AI to
update both the label and the comparison logic. The three additional features
(price explanation, similar cars, What If comparison) were my own ideas which I
described and then reviewed after implementation.

### Report and Slides

AI helped structure the Quarto report and slides based on the analysis already
completed. The content — the numbers, the interpretations, the model comparisons —
all came from the actual results. AI helped me turn those results into readable
text and a consistent format. I reviewed every section of the report and corrected
places where the explanation did not match my understanding.

---

## What Worked Well

**Speed of implementation.** Tasks that would have taken me many hours — writing a
full training pipeline, setting up a Streamlit app, deploying to GitHub — were
completed much faster with AI assistance. This left more time to focus on
understanding the results rather than fighting with syntax errors.

**Explaining concepts.** When I did not understand something (for example, why
`VarianceThreshold` is leakage-safe, or what MAE versus RMSE actually measures),
I could ask the AI and get a plain-language explanation immediately. This was more
useful than reading documentation because I could ask follow-up questions.

**Catching errors.** The AI caught several mistakes before they became problems —
for example, a bug in the row-count audit where filters were applied simultaneously
instead of sequentially, and a `TypeError` in the metadata JSON code during training.

---

## What Did Not Work Well / Limitations

**AI cannot replace understanding.** There were moments when AI produced correct
code that I did not fully understand. I had to stop and ask for an explanation
before continuing, because if I had submitted work I could not explain, I would
not be able to answer questions about it. AI is a tool for faster implementation,
not a substitute for learning.

**Decisions still require the student.** AI does not know the course requirements,
the teacher's preferences, or what makes a good scope decision for this specific
dataset. Every significant decision — what price range to include, which metric to
prioritise, what to put on a slide — had to come from me, not from the AI.

**Deployment problems required real troubleshooting.** When the GitHub push failed
with an HTTP 400 error, or when the app was not appearing on Streamlit Cloud, the
AI could suggest fixes but I had to execute them, understand what went wrong, and
verify that the fix worked. Some of these steps required running commands in the
terminal and reading error messages carefully.

**The AI can be overconfident.** A few times, the AI suggested a solution that
seemed reasonable but was not quite right for my specific situation. I learned to
always test the output rather than assume it was correct.

---

## What I Learned from Using AI This Way

Working with AI on a project this size taught me that **the quality of the result
depends on the quality of the instructions**. Vague prompts produced vague results.
When I was specific about what I wanted — "filter the fuel type dropdown to only
show fuels that actually exist for this brand and model in the dataset" — the
output was useful. When I was vague — "improve the sidebar" — I had to do more
reviewing and correcting.

I also learned that understanding the code is more important when using AI, not
less. Because the AI can generate code very quickly, I had to read it more carefully
to check that it was doing what I intended. Copying code without understanding it
would have been easy, but it would have made the project impossible to explain or
defend.

Finally, AI made me more confident about attempting things I would have avoided
because of the time cost — for example, writing a full Quarto report with embedded
figures, or adding a feature importance explanation section to the app. These felt
achievable because the implementation barrier was lower, but the conceptual
understanding still had to come from me.

---

## Summary

AI was a significant part of how this project was built. It accelerated
implementation, caught errors, and helped me understand concepts by explaining them
in plain language. However, every major decision was mine, every result was
verified, and the conceptual understanding of what the models do, why the pipeline
is designed the way it is, and what the results mean was developed through the
process of working through the project — not handed to me by the AI.

The most honest description of how I worked: I directed the AI, reviewed its output,
corrected it when it was wrong, and made sure I could explain everything it
produced before moving on.
