---
title: "Germany Used Cars 2023 — ML Project Deliverables"
author: "Anja Kadic"
date: "July 2026"
---

# D5 — Executive Summary

## The Problem

Buying or selling a used car in Germany is difficult because prices are not transparent.
Two sellers can list identical cars at very different prices, and buyers have no easy
way to know whether a listing is fairly priced. This project builds a data-driven
tool that gives anyone an objective, instant price estimate for a used car based
on real market data.

## The Data

The project uses the **Germany Used Cars 2023** dataset — a collection of
**251,079 real car listings** scraped from AutoScout24, Germany's largest used-car
marketplace. Each listing contains the car's brand, model, year, fuel type,
transmission, horsepower, and mileage, along with the asking price.

After removing listings outside the general used-car market (pre-2000 vehicles,
prices above €80,000, new-condition cars, and entries with data errors),
**240,017 listings** remained for training and testing.

## What Was Built

Two machine learning models were trained on this data:

**Price segment classifier** — predicts whether a car belongs in the HIGH or LOW
price segment. The dividing line is €19,486, which is the median price of all
training listings. This model helps a buyer or seller quickly understand whether
a car is above or below the typical market price.

**Price regressor** — predicts the exact expected listing price in euros. A user
enters seven pieces of information about a car and receives an estimated price
based on patterns learned from 192,000 real listings.

Both models were trained using a technique called **XGBoost**, which builds many
small decision trees and combines them to capture complex patterns in the data.
Four models of increasing complexity were tested for each task, and XGBoost
consistently outperformed the simpler alternatives.

## Key Results

The models were tested on **48,004 listings that were never used during training**
to ensure the reported performance is honest and reflects real-world accuracy.

| What was predicted | How accurate |
|---|---|
| Price segment (HIGH or LOW) | Correct in **93.4%** of cases |
| Exact listing price | Average error of **€2,602** per car |

To put these numbers in context: a simple baseline that always guesses the average
price makes an average error of **€10,915**. The trained model reduces that error
by more than 75%. For the price segment task, a model that always guesses HIGH
would be correct only 50% of the time — the trained model reaches 93.4%.

The analysis also revealed that used-car pricing follows a largely **additive
pattern**: a car's price is mainly determined by which brand it is, how old it is,
how many kilometres it has, and how powerful it is. These four factors combined
explain most of the price variation across the dataset.

## The Application

The results are deployed in a **live web application** accessible at:

**https://germany-used-cars-ml.streamlit.app/**

Any user can visit the app and enter a car's details to receive:

- An estimated market price in euros
- A classification of HIGH or LOW price segment with a confidence percentage
- A plain-language explanation of why the model predicted that price
- A list of the eight most similar real cars from the dataset with their actual prices
- A side-by-side comparison of two different cars

The application works on any device with a browser and requires no installation.

## Limitations

The models are designed specifically for general-market used cars listed in
Germany in 2023. They are not suitable for classic or collector cars, brand-new
vehicles, ultra-luxury cars above €80,000, or cars in other countries where
pricing dynamics differ.

The predictions reflect **asking prices**, not final sale prices. Real transactions
typically settle 5–15% below the listed price.

The model was trained on 2023 data. Market conditions change over time due to
fuel prices, interest rates, new model releases, and shifts in consumer demand,
so accuracy may decrease for listings from significantly different time periods.

## Summary

This project demonstrates that machine learning trained on real listing data can
predict used-car prices with meaningful accuracy using only basic information that
any buyer or seller already knows. The live application makes these predictions
accessible to anyone and provides transparent explanations so users understand
not just the prediction, but why it was made.

---

\newpage

# D3 — AI Workflow Reflection

## Overview

This document reflects honestly on how I used AI assistance throughout this machine
learning project — what I used it for, where it helped, where it had limitations,
and what I learned from the experience of working alongside an AI tool.

The AI tool I used was **Claude Code** (Anthropic), an AI assistant integrated
into the terminal that can read files, write code, and run commands.

## How I Used AI — Task by Task

### Data Cleaning

I used AI to help structure the twelve sequential cleaning steps in `src/clean.py`.
The key decision — that the filters had to be applied **one at a time** and not all
at once — came from understanding what "sequential" means for row-count accounting.
When an early version of the code applied all masks simultaneously, the row counts
were wrong. I asked the AI to explain why, and once I understood the problem I was
able to verify the corrected version myself.

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

I used AI to write the model ladder (Dummy → Logistic/Ridge → Decision Tree →
XGBoost). AI suggested using `RandomizedSearchCV` instead of `GridSearchCV` for
XGBoost tuning to reduce training time, which was practical advice I verified by
comparing the two approaches.

The most important thing I had to understand myself was **why F1_macro is the right
metric and F1_binary is misleading**. The Dummy classifier achieves F1_binary = 0.667
by always predicting HIGH — this looks good but is completely uninformative.
F1_macro = 0.333 is the honest number. I had to be able to explain this distinction
clearly, and the AI helped me check my understanding by asking it follow-up questions.

### Leakage Prevention

Preventing data leakage was the most important correctness requirement in the project.
AI helped me structure the pipeline so that the OneHotEncoder, StandardScaler, and
price segment threshold are all fitted on training data only. I tested my
understanding by asking: "What would happen if we computed the threshold on the
full dataset?" — and the AI explained the consequence clearly enough that I could
then explain it in my own words.

### Streamlit App

AI wrote most of the initial Streamlit app code based on my instructions about
what the interface should look like and what features it should have. I reviewed
every section, tested it locally, and gave feedback when something did not work
the way I expected. The three additional features — price explanation, similar
cars, and the What If comparison — were my own ideas which I described and then
reviewed after implementation.

### Report and Slides

AI helped structure the Quarto report and slides based on the analysis already
completed. The content — the numbers, the interpretations, the model comparisons —
all came from the actual results. AI helped me turn those results into readable
text and a consistent format. I reviewed every section and corrected places where
the explanation did not match my understanding.

## What Worked Well

**Speed of implementation.** Tasks that would have taken me many hours — writing a
full training pipeline, setting up a Streamlit app, deploying to GitHub — were
completed much faster with AI assistance. This left more time to focus on
understanding the results rather than fighting with syntax errors.

**Explaining concepts.** When I did not understand something — for example, why
VarianceThreshold is leakage-safe, or what MAE versus RMSE actually measures — I
could ask the AI and get a plain-language explanation immediately. I could also
ask follow-up questions until I genuinely understood.

**Catching errors.** The AI caught several mistakes before they became problems,
including a bug in the row-count audit and a TypeError in the training script.

## What Did Not Work Well / Limitations

**AI cannot replace understanding.** There were moments when AI produced correct
code that I did not fully understand. I had to stop and ask for an explanation
before continuing, because if I had submitted work I could not explain, I would
not be able to answer questions about it.

**Decisions still require the student.** AI does not know the course requirements,
the teacher's preferences, or what makes a good scope decision for this specific
dataset. Every significant decision had to come from me.

**Deployment problems required real troubleshooting.** When the GitHub push failed
or the app was not appearing on Streamlit Cloud, the AI could suggest fixes but I
had to execute them, understand what went wrong, and verify that the fix worked.

**The AI can be overconfident.** A few times, the AI suggested a solution that was
not quite right for my specific situation. I learned to always test the output
rather than assume it was correct.

## What I Learned from Using AI This Way

Working with AI on a project this size taught me that **the quality of the result
depends on the quality of the instructions**. Vague prompts produced vague results.
When I was specific about what I wanted, the output was useful and accurate.

I also learned that understanding the code is more important when using AI, not
less. Because the AI can generate code very quickly, I had to read it more
carefully to check that it was doing what I intended.

Finally, AI made me more confident about attempting things I would have avoided
because of the time cost — writing a full Quarto report, adding explanation
features to the app. These felt achievable because the implementation barrier was
lower, but the conceptual understanding still had to come from me.

## Summary

AI was a significant part of how this project was built. It accelerated
implementation, caught errors, and helped me understand concepts through plain-
language explanations. However, every major decision was mine, every result was
verified, and the conceptual understanding of what the models do and what the
results mean was developed through the process of working through the project.

The most honest description of how I worked: I directed the AI, reviewed its
output, corrected it when it was wrong, and made sure I could explain everything
it produced before moving on.
