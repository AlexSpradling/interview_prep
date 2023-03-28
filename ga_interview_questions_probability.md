# TECHNICAL - PROBABILITY/LOGIC

### Suppose that we are in a 100-story building with exactly two eggs. We wish to know which story is the highest story from which I can drop an egg without breaking the egg. If I drop an egg to the ground and it breaks, then we can assume that dropping an egg from every floor above that will also result in a broken egg. Similarly, if I drop an egg to the ground and it does not break, then we can assume that dropping an egg from every floor below that will also not break the egg. If we drop an egg and it breaks, we cannot drop that egg again. If we drop an egg and it does not break, then we can drop that egg again. What strategy should we use to discover the highest floor from which we can safely drop eggs, but that minimizes the worst-case scenario total number of drops needed? **(repeat)** [Helpful link.](https://www.geeksforgeeks.org/puzzle-set-35-2-eggs-and-100-floors/)


The goal is to minimize the number of drops in the worst-case scenario. Let’s say we make our first attempt on the x-th floor. If the egg breaks, we have (x-1) remaining drops to check floors 1 to (x-1) with the second egg. If it doesn’t break, we move up (x-1) floors and drop the first egg again. This way, if the first egg breaks at any point, we have enough remaining drops to check all floors below it with the second egg.

The total number of floors we can check with this strategy is given by the equation: $x + (x-1) + (x-2) + … + 1 = 100$. Solving this equation for x gives us $x = 14$$. This means that if we start at floor 14 and move up 13 floors each time the first egg doesn’t break, we will find the highest floor from which we can safely drop eggs within a maximum of **14** drops.


---
### If a coin was tossed 10 times and it came up heads 70% of the time, what is the probability that it is a biased coin?

**Note**: *I don't know how to answer this.

We can use the [[Binomial Distribution]] to calculate the probability of getting 7, 8 , 9, and 10 out of 10 heads and sum their probabilities to get the p-value of getting heads 70% of the time/The p-value is the probability of observing 7 or more heads in 10 tosses if the coin were fair.

$$
\begin{align*}
{10\choose 7} \times 0.5^{7}\times 1-0.5^{10-7} &= .117\\

{10\choose 8} \times 0.5^{8}\times 1-0.5^{10-8} &= .044\\

{10\choose 9} \times 0.5^{9}\times 1-0.5^{10-9} &= .010\\

{10\choose 10} \times 0.5^{10}\times 1-0.5^{10-10} &= .001\\
\\
p = .117 + .044 + .010 + .001 &= .17\\\\
\end{align*}
$$

With a p-value of .17 we can't assume the coin is biased. 

---
### What are the chances that, in a group of 4 people, all four people are born in different seasons?

Assuming that the seasons are equally likely and that the birthdays of the four people are independent, the probability can be calculated like this:

1. The first person an be born in any season, so the probability they are born in a unique season is 1. 
2. The second person must be born in a different season than the first. So the probability they area born in a different season is $\frac{3}{4}$.
3. The third person must be born in a different season than the first and the second so the probability they are born in a different season is $\frac{2}{4}$.
4. The fourth person must be born in a different season than everyone else so the probability they are born in a different season is $\frac{1}{4}$. 

We multiply the probabilities:
$$
1 \times \frac{3}{4}\times \frac{2}{4}\times \frac{1}{4}= .09375
$$


---
### Suppose you buy `n` tickets and 20% of tickets are fraudulent. What is the expected number of fraudulent tickets? What is the probability that none of the tickets are fraudulent?

The expected value of a binomial random value is $E(x) = p \times n$. 

So we would expect $.2 n$ tickets to be fraudulent and $.8n$ to not be fraudulent.

---
### There are 24 balls: 12 red and 12 black. If you draw two balls, what is the probability of drawing the same color ball both times?

$$
\left(\frac{12}{24}\times \frac{11}{23}\right)\times 2  \textrm{(for r and b balls)} = .47
$$

---
### Say you wanted to roll a fair 6-sided die but you didn't have one available. How could you mimic the results with a fair coin instead?

Flip the coin 3 times and convert the results into a binary string where heads = 0 and tails = 1. For example THT = 010. To convert from binary string to decimal, raise 2 to the power of each binary string, so:
$$
2^{0}+ 2^{1}+ 2^{0}= 2
$$
Then, we add $1$ to the decimal to get a number between 1 and 6 with equal probability. 

The only case where this breaks down is where we flip TTT, which is:

$$
2^{1}+ 2^{1}+ 2^{1}= 6
$$
PLUS $1$ to convert to decimal so we have $7$, and will need to start over. 

---
### You are currently in a game of tennis, where each person has won one point each. If you win each point 60% of the time and each point is independent, what is your probability of winning the game? (In tennis, a game is played until one player wins. A player wins when they have greater than or equal to 4 points and at least 2 more points than their opponent.)


---
### There are four baseball teams. The Boston Red Sox is facing the New York Yankees in one series. The Los Angeles Dodgers are facing the San Francisco Giants in another series. Each pair will play five games against each other. The first team to win 3 games wins that series. The winners of each series will play one another in the World Series, which has the same format. For each game, the home team has a 2/3 chance of winning while the away team has a 1/3 chance of winning. In the first series, the Red Sox will be the home team in games 1, 2, and 5 and the away team in games 3 and 4. In the second series, the Dodgers will be the home team in games 1, 2, and 5 and the away team in games 3 and 4. In the World Series, the winner of the first series (either the Red Sox or the Yankees) will be the home team in games 1, 2, and 5 and the away team in games 3 and 4. What is the probability that the Boston Red Sox defeat the San Francisco Giants by 3 games to 2 in the World Series?

I'll answer this question at some point in my life but I'm not interested in working at a place that asks me a question like that. 

---
### Imagine there are a 100 people in line to board a plane that seats 100. The first person in line realizes they lost their boarding pass, so when they board, they pick a random seat instead. Every person that boards the plane afterward will take their assigned seat or, if that seat is taken, a random seat instead. What is the probability that the last person getting on the plane (person 100) sits in their assigned seat? (Solution: https://math.stackexchange.com/questions/5595/taking-seats-on-a-plane)

Suppose whenever someone finds their seat taken, they politely evict the squatter and take their seat. In this case, the first passenger (Alice, who lost her boarding pass) keeps getting evicted (and choosing a new random seat) until, by the time everyone else has boarded, she has been forced by a process of elimination into her correct seat.

This process is the same as the original process except for the identities of the people in the seats, so the probability of the last boarder finding their seat occupied is the same.

When the last boarder boards, Alice is either in her own seat or in the last boarder's seat, which have both looked exactly the same (i.e. empty) to her up to now, so there is no way poor Alice could be more likely to choose one than the other. 

So, the probability is $\frac{1}{2}$

---
### You have the option to throw a die **up to** three times. You will earn the face value of the die. You have the option to stop after each throw and walk away with the money earned. The earnings are not additive. (That is, you only get the money earned from the most recent dice roll.) What is the expected payoff of this game? (Solution: https://math.stackexchange.com/questions/179534/the-expected-payoff-of-a-dice-game)

(Theory of optimal stopping for Markov Chains)

In a 1 roll game, the expected value is the average of $1, 2, 3, 4, 5, 6 = 3.5$

If we have 2 rolls, we have these options:

1. If we roll 4, 5, or 6 we should keep the roll as this is higher than the expected value of a 1 roll game. If we roll a 1, 2, or 3, we re-roll since the expected payout of a 1-roll game is 3.5 so now we have .$5 (5) + .5 (3.5) = 4.25$
2. In a 3 roll game, if we roll a 5 or a 5, we keep it, but if we roll a 4, we take a re-roll since a 2-roll game has an expected value of 4.25.

So the final expected payout is:

$$
\frac{1}{3}(5.5) + \frac{2}{3}(2.45) = 4.66
$$

---
### You're about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it's raining. Each of your friends tells you the truth with probability 2/3 (and therefore will lie 1/3 of the time). All 3 friends tell you that "Yes" it is raining. Based on historical evidence, there is a 25% chance it is raining at any given time in Seattle. What is the probability that it's actually raining in Seattle right now, given that all 3 of your friends tell you that it is raining?


This is a [[Bayes' Theorem]] probability problem. We'll use this flavor of Bayes' rule to solve:

$$
P(A|B) = \frac{P(B|A)\times P(A)}{P(B|A)\times P(A) + P(B|notA) \times P(notA)}
$$

**Breaking It Down**

$P(A|B)$ is the probability that it is raining, given that all 3 friends tell you it is raining. 

$P(B|A)$ is the probability your friends are telling you it is raining given that it is raining. Which means they are telling the truth, which they do $1/3$ of the time. We can find this conditional probability like this:

$$
\begin{align*}\\\\

P(B) &= \textrm{prob of truth}\\\\

P(A) &= \textrm{prob of rain}\\\\


P(B|A) &= \frac{P(A)\times P(B))}{P(A)}\\
\textrm{It is raining and they are telling the truth, so P(B) is .66}\\
P(B|A) &= \frac{\frac{2}{3}^{3}\times .25}{.25} = .296\\
\end{align*}\\
$$
$P(B|notA)$ is the probability your friends are telling you it is raining given that it is *not* raining, which means they are lying -- which they do $1/3$ of the time. We can find this 
$$
\begin{align*}
P(B) &= \textrm{prob lying}\\\\

P(notA) &= \textrm{prob not raining}\\\\
P(B|notA) &= \frac{\frac{1}{3}^{3}\times .75}{.75}
\end{align*}

$$

**Plugging everything into the original formula..

$$
\begin{align*}
P(A|B) &= \frac{P(B|A)\times P(A)}{P(B|A)\times P(A) + P(B|\neg A)\times P(\neg A)}\\\\
&= \frac{(2/3)^3 \times 0.25}{(2/3)^3 \times 0.25 + (1/3)^3 \times 0.75}\\\\
&= \frac{8}{11}\\\\
&\approx \boxed{0.727}
\end{align*}
$$



---
### Suppose you throw three dice. What is the probability that your rolls are in increasing order?

There are $6^{3} = 216$ ways to throw 3 dice, and  ${6 \choose 3} = 20$ ways of choosing 3 numbers from the rolls. The numbers can only be in ascending order 1 way so the probability is $\frac{20}{216}=.0926$

---
### Given a sample of apple prices in DC and a sample of apple prices in NYC, how might you try to estimate the probability that a randomly selected apple in DC is more expensive than a randomly selected apple in NYC?

One way to estimate the probability that a randomly selected apple in DC is more expensive than a randomly selected apple in NYC is to use the samples of apple prices from both cities to calculate the proportion of times that an apple price from the DC sample is greater than an apple price from the NYC sample.
