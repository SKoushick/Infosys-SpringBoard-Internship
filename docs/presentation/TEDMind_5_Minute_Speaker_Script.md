# TEDMind: 5-Minute Master Speaker Script
### Infosys Springboard Internship 7.0 — Final Project Review Presentation

**Author:** S Koushick  
**Platform:** TEDMind (*Discover. Focus. Learn. Grow.*)  
**Total Pacing:** ~135 Words Per Minute  
**Total Duration:** 5 Minutes 15 Seconds  

---

> [!TIP]
> **Speaker Delivery Advice:** Speak at a steady, natural cadence. Emphasize transition sentences as verbal bridges before advancing each slide.

---

## Slide 1: Title Slide — Welcome & Project Overview
**Target Timing:** `0:00 – 0:30 (30 seconds)`  

Good morning, respected mentors and evaluation panel. My name is S Koushick, and today I am privileged to present TEDMind: an AI-powered personalized TED Talks recommendation and goal-oriented learning platform, developed as the capstone engineering deliverable for the Infosys Springboard Internship 7.0.

Our guiding philosophy is simple: 'Discover. Focus. Learn. Grow.' TEDMind bridges the gap between passive video streaming and intentional educational pathways by uniting classical lexical retrieval, dense neural semantic transformers, and algorithmic diversity re-ranking across 2,550 presentations from the global TED catalog.

> **[Slide Transition Cue]:** *Let us examine the fundamental real-world challenge that motivated this system.*

---

## Slide 2: Problem Statement — Friction in Self-Directed Learning
**Target Timing:** `0:30 – 1:00 (30 seconds)`  

Online learners face severe discovery friction. In a sprawling catalog like TED, viewers experience severe choice paralysis—navigating over 2,500 interdisciplinary talks without a structured path.

Furthermore, traditional search suffers from vocabulary mismatch: keyword search retrieves only literal word matches, completely missing relevant presentations that express the same concept using alternative vocabulary. Finally, standard content-based algorithms trap users in narrow filter bubbles by repeatedly recommending near-identical topics, with zero evaluation of whether a talk is foundational or advanced.

> **[Slide Transition Cue]:** *This motivated us to build TEDMind as a multi-signal, goal-oriented solution.*

---

## Slide 3: Project Objectives & Solution Overview
**Target Timing:** `1:00 – 1:30 (30 seconds)`  

TEDMind is designed for students, researchers, and professional lifelong learners. Instead of treating recommendations as isolated entertainment suggestions, TEDMind provides three core innovations:

First, multi-signal blending: we combine exact keyword matching with deep semantic context. Second, diversity optimization via Maximal Marginal Relevance, ensuring that recommended slates are broad and serendipitous rather than repetitive. And third, transparent Explainable AI badges, providing clear reasons for every suggestion so learners understand why a talk was recommended.

> **[Slide Transition Cue]:** *To achieve this, we began with rigorous data collection and preprocessing in Modules 1 and 2.*

---

## Slide 4: Dataset & Preprocessing (Modules 1 & 2)
**Target Timing:** `1:30 – 2:00 (30 seconds)`  

Our canonical dataset encompasses 2,550 curated presentations published between 2006 and 2020, combining rich metadata with full speech transcripts. In Module 1, we validated complete catalog integrity, resolving missing IDs and pruning broken cross-references.

In Module 2, we conducted text sanitization: we engineered regular expressions to strip stage cues such as laughter and applause from transcripts, normalized Unicode entities, and for talks lacking transcripts, we synthesized a high-density 'content soup' weighting titles, speakers, tags, and descriptions. This ensured zero catalog talks had to be discarded.

> **[Slide Transition Cue]:** *With a pristine text corpus, we architected our dual-space vector representation in Module 3.*

---

## Slide 5: Feature Engineering & Vectorization (Module 3)
**Target Timing:** `2:00 – 2:30 (30 seconds)`  

Recommenders often fail when relying on a single vector space. In Module 3, we engineered a dual-space architecture.

In the sparse lexical space, a sublinear TF-IDF vectorizer extracts 44,613 unigram and bigram features. This guarantees laser precision on specialized technical terms, acronyms, and proper speaker names.

In parallel, in the dense semantic space, we utilized Sentence-BERT using the all-MiniLM-L6-v2 transformer to project talks into 384 continuous dimensions. This captures nuanced philosophical concepts, thematic synonyms, and conceptual paraphrasing that keyword search misses completely.

> **[Slide Transition Cue]:** *These two spaces feed directly into our multi-signal recommendation engine in Module 4.*

---

## Slide 6: Recommendation Engine & Diversification (Module 4)
**Target Timing:** `2:30 – 3:15 (45 seconds)`  

Our recommendation engine blends four distinct signals into a unified hybrid score: 35% lexical TF-IDF, 45% dense Sentence-BERT similarity, 15% curated taxonomy tag overlap, and 5% speaker affinity, with dynamic exclusion of user-disliked talks.

To solve the filter-bubble dilemma, we applied Maximal Marginal Relevance—or MMR—with a tuned lambda of 0.70. MMR penalizes redundant candidate talks that are overly similar to presentations already selected for the slate. This delivers an outstanding 20.6% boost in intra-list diversity while preserving ranking precision, all computed in sub-60 millisecond latency.

> **[Slide Transition Cue]:** *Let us examine our empirical benchmark results evaluated in Module 5.*

---

## Slide 7: Empirical Evaluation & Results (Module 5)
**Target Timing:** `3:15 – 3:45 (30 seconds)`  

In Module 5, we benchmarked our models across 100 standardized evaluation queries under identical conditions. As shown in our scorecard, the Hybrid Multi-Signal model achieved the highest overall ranking accuracy, with an NDCG@10 of 0.0076 and MAP@10 of 0.0035, outperforming standalone lexical and semantic baselines.

Crucially, introducing MMR expanded Intra-List Diversity to 0.6953—a 20.6% increase—while expanding catalog coverage to 54.16%. In accordance with scientific integrity, we note that ground truth is computed against TED's curated companion talk proxy, and qualitative human evaluation is formally cataloged as pending.

> **[Slide Transition Cue]:** *Now, let us view how this intelligence is delivered in our production web application in Module 6.*

---

## Slide 8: Web Application Architecture & UI (Module 6)
**Target Timing:** `3:45 – 4:15 (30 seconds)`  

Module 6 delivers a full-stack Django 5.2 application styled with modern Tailwind CSS. The slide displays genuine live screenshots captured from our running server.

Learners can explore all 2,550 presentations in our catalog explorer, dive into talk details with synchronized speech transcripts, and interact with our Recommendation Studio by providing seed talks, curiosity prompts, or topic themes. Guest sessions seamlessly migrate bookmarks upon user login, and authentication gateways securely protect private collections and administrative controls.

> **[Slide Transition Cue]:** *Before concluding, let us review our engineering disclosures and Phase 2 progress.*

---

## Slide 9: Limitations, Scientific Integrity & Phase 2
**Target Timing:** `4:15 – 4:45 (30 seconds)`  

We maintain strict engineering transparency: collaborative filtering is accurately unsupported because public TED datasets lack individual user interaction IDs. Furthermore, similarity scores represent geometric vector distances rather than calibrated probabilities.

In Phase 2, we progressed into pedagogical tracking. We implemented an automated complexity estimator profiling readability, lexical richness, and speaking pace into foundational, intermediate, and specialized stages. We also built transactional path deletion services that guarantee learner watch-time and ratings are never orphaned or lost.

> **[Slide Transition Cue]:** *To summarize our journey and practical contributions in Module 7.*

---

## Slide 10: Conclusion & Practical Impact (Module 7)
**Target Timing:** `4:45 – 5:15 (30 seconds)`  

In conclusion, TEDMind represents an end-to-end engineering achievement across all seven Infosys Springboard modules. With 110 automated tests passing at a 100% rate, robust database referential integrity, and complete technical documentation, we have demonstrated how recommender systems can evolve from passive entertainment feeds into intentional, diversity-aware educational platforms.

I extend my sincere gratitude to our Infosys Springboard mentors. Thank you, and I look forward to your questions.

> **[Slide Transition Cue]:** *End of Presentation — Q&A Session.*

---
