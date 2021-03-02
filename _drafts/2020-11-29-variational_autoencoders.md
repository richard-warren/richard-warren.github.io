---
title: 'variational autoencoders'
date: 2020-11-29
permalink: /blog/vae/
tags:
  - machine learning
  - probability
read_time: false
---


description

{% include toc %}
<br>



- low dim data representation
  - compression
  - dots on line example
  - two image example
  - goal: find good low dimensional representation
- start with ending
  - this is inference problem, but let's treat like optimization
  - reconstruction loss term
  - kl divergence regularization
- prob perspective  
  - existence of joint prob distribution
  - see data, want to know latent state
  - don't just want z for every x, but prob dist over z, p(z|x)
  - derive bayes rule
  - explain prior, evidence, likelihood, posterior
  - expand p(x) and explain problem!
- solution
  - find q(z) that approximates p(z|x)
  - minimize kl divergence
  - explain kl divergence
  - work out formula for p(x)
  - we can't minize kl p bc p is unknown, but know we can maximize this other term!
  - but what is this other term?
- evidence lower bound
  - work for p(x) to evidence lower bound using jensen's inequality
  - now we know what that term was in previous equation!
  - getting best approximation of posterior is equivalent to finding params that maximize elbo
- try it out
