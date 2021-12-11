To run script, cd into the root directory of the project & type the following command:
```angular2html
./script.sh
```

# Part 2 & 3
We have merged pt 2 and 3 of the HMM, by generating a list of i outputs and sorting them in order, where i = 1 in pt 2
and 5 in pt 3!

# Part 4
Now, based on the training and development set, think of a better design for developing an im- proved sentiment analysis
system for tweets using any model you like. Please explain clearly the model/method that you used for designing the new
system. We will check your code and may call you for an interview if we have questions about your code. Please run your
system on the development set RU/dev.in and ES/dev.in. Write your outputs to RU/dev.p4.out and ES/dev.p4.out. Report the
precision, recall and F scores of your new systems for these two languages.
(10 points)

A disadvantage of the HMM is that it only remembers the previous state. In real life, when we interpret the sentiment of
a word we use much more of the surrounding context to make our judgements.

Hence we extend the HMM by hypothesizing that each new state does not solely depend on the previous state, but also
the state before it. In this model, instead of having transmission parameters $t(y_n | y_{n-1})$ we have a new transmission
parameter $t(y_n | y_{n-1}, y_{n-2})$.

To find the tag sequence that gives us the maximum score, we extend Viterbi's algorithm to keep track of the maximum
score so far given the current tag and the previous tag. Letting this be $b_{i, y_i, y_{i-1}}$, we have the recurrence

\[
b_{i, y_i, y_{i-1}} = \max_{y_{i-2}} b_{i-1, y_{i-1}, y_{i-2}} e(x_i | y_i) t(y_i | y_{i-1}, y_{i-2})
\]

We found that by doing this, our Sentiment F scores improve by about 0.2 in their evaluation of both the ES and RU development sets.

```
# ES

#Entity in gold data: 255
#Entity in prediction: 215

#Correct Entity : 139
Entity  precision: 0.6465
Entity  recall: 0.5451
Entity  F: 0.5915

#Correct Sentiment : 112
Sentiment  precision: 0.5209
Sentiment  recall: 0.4392
Sentiment  F: 0.4766

# RU

#Entity in gold data: 461
#Entity in prediction: 353

#Correct Entity : 239
Entity  precision: 0.6771
Entity  recall: 0.5184
Entity  F: 0.5872

#Correct Sentiment : 163
Sentiment  precision: 0.4618
Sentiment  recall: 0.3536
Sentiment  F: 0.4005
```

We have attempted to extend this model such that our transitions depend on the previous 3 states instead, but
we have found that this does not meaningfully affect our F scores.