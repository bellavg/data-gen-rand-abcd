# Scope and Delimitations: exemplars and a recommendation

Working note. Not part of the thesis build.

## Her current section

`sections/1-introduction/1.2-research-questions.tex`, `\subsection{Scope and Delimitations}`,
47 lines, roughly 300 to 350 words of prose once LaTeX markup is stripped out. One opening
paragraph (four sentences: optimizability as instrument not objective, the delimitation
versus limitation split, a pointer to where limitations live instead) followed by six
bold-lead-in paragraph items: target synthesis algorithm, definition of optimizability,
graph scale, reduction families, model architecture, prediction rather than script
generation. Each item runs two to three sentences: a claim, a reason, usually a forward
reference. Already a second pass, 61 lines compressed to 47.

## Five verified examples

Every excerpt below was checked against the source PDF directly, not against a summary or a
template site. The sample is small and skews toward one repository, KTH's DiVA, for four of
five, because that is where full text was reliably fetchable. Treat the pattern below as
suggestive, not as a survey.

**1. Roni Henareh, "Fine-Tuning Small Open-Weight LLMs for Cybersecurity," KTH degree
project (MSc), 2026.**
[PDF](https://kth.diva-portal.org/smash/get/diva2:2060365/FULLTEXT01.pdf)
Section 1.6, two sentences total. One: "Collecting data for training is also out of scope,"
then the same sentence names the substitute, an existing Huggingface dataset used instead of
collecting one. Works because the exclusion states what filled the gap, not only the gap.
Length: two sentences, about 30 words.

**2. Delwende Eliane Birba, "A Comparative Study of Data Splitting Algorithms for Machine
Learning Model Selection," KTH degree project (MSc), 2020.**
[PDF](https://www.diva-portal.org/smash/get/diva2:1506870/FULLTEXT01.pdf)
Section 1.6, three bullets. One reads: "efficiency is not evaluated in this thesis as it is
not the scope." Works because the reason is welded to the claim in one clause ("as..."),
not argued across a second sentence. Length: three bullets, about 55 words.

**3. Amirhossein Namazi, "Evaluation of Pruning Algorithms for Activity Recognition on
Embedded Machine Learning," KTH master's thesis, 2023.**
[PDF](https://www.diva-portal.org/smash/get/diva2:1851707/FULLTEXT01.pdf)
Section 1.5, prose: "The models are trained on a set of finite datasets that are rather
small." Included as a caution rather than a model to copy: the same paragraph later says
"another limitation are the architectures," blurring delimitation into limitation inside a
section headed only "Delimitations." Length: one paragraph, about 75 words.

**4. Ifrah Tariq, "Biologically Interpretable Representation Learning for Mechanistic
Insights into Cancer Immunotherapy Resistance," PhD dissertation, MIT, 2025.**
[PDF](https://dspace.mit.edu/bitstream/handle/1721.1/164583/tariq-ifrah-phd-csb-2025-thesis.pdf)
Section 1.9, "Assumptions, Limitations, and Delimitations," all three named and separated in
one place. The delimitations block: "The study is focused specifically on resistance
mechanisms to ICI therapy." Works because every bullet is tagged by category first (Scope,
Data Modalities, Cancer Types, Computational Model), then stated in one line. The only
PhD-level example found, and still four tagged bullets, one line each. Length: four bullets,
about 70 words for the delimitations block alone.

**5. Rasmus Craelius, "Enhancing Fraud Detection with Graph-Based Machine Learning," KTH
degree project (MSc), 2025.**
[PDF](https://www.diva-portal.org/smash/get/diva2:1998930/FULLTEXT01.pdf)
Section 1.5, three bullets, on a graph convolutional model directly. Given the dataset's
anonymized features, the section reads: "feature importance is not part of the model
comparison or evaluation." Works for the same reason as Birba: cause, then consequence, one
sentence. Length: three bullets, about 60 words.

## Patterns

- **Location.** Chapter 1 in every case, always its own numbered subsection: 1.5 in three of
  the five, 1.6 in one, 1.9 in the PhD dissertation, where assumptions and thesis structure
  are separately numbered around it. Matches where hers already sits.
- **Delimitations versus limitations, kept apart.** Cleanest in the strongest example
  (Tariq): delimitations are choices fixed in advance, limitations are what the work exposed,
  assumptions are a third, separate category. The weakest example (Namazi) fails exactly
  here, and reads worse for it: a sentence starting "another limitation..." inside a section
  headed "Delimitations" forces the reader to reclassify it mid-paragraph. Her thesis already
  keeps this split: the scope subsection covers choices fixed in advance, and
  `sec:discussion:limitations` is reserved for what surfaced during the work. That split is
  standard practice, not an idiosyncratic choice, and the strongest exemplar found treats it
  as load-bearing enough to name all three categories explicitly rather than trusting a
  section heading alone to carry the distinction.
- **Prose or list.** Split evenly, three bulleted (Birba, Tariq, Craelius), two prose
  (Henareh, Namazi). Format does not predict length. The bulleted examples are not shorter
  per item than the prose ones.
- **Length.** Two sentences to four bullets. None of the five ran past about 80 words for the
  delimitations content itself, PhD dissertation included. Detail that would make an item
  longer is pushed to wherever that choice is actually argued, not kept in the section
  labelled delimitations.
- **One clause per exclusion, reason attached.** Four of the five weld the reason to the
  claim inside a single sentence: "X is out of scope, [reason]" or "not evaluated... as it is
  not the scope." None spend a separate sentence justifying and a third restating impact.
  This is the biggest lever for compression, and the one her current draft has not yet
  pulled: her six items each carry the reason and the forward reference as separate clauses
  or sentences rather than one.

## Recommendation

**Target: 15 to 20 lines, four to five items, one sentence each.** Roughly a 60 percent cut
from the current 47 lines, in line with every verified example above, PhD dissertation
included.

**Keep the bold-lead-in paragraph format, do not switch to bullets.** It already matches how
`sec:discussion:limitations` is set (`\emph{Label.}` then prose), and the two sections are
explicitly meant to be read as a pair (the opening paragraph says so directly). Changing
format here would cost more in cross-section consistency than it buys in compactness. The
fix is sentence count per item, not the container.

**Two merge candidates**, since every exemplar states one exclusion per item and two pairs
of her six currently split one exclusion across two items:
- "Definition of optimizability" and "Prediction rather than script generation" both bound
  what the target $y$ captures: node count only, not power or delay, and a ranking of
  circuits under one script, not a ranking of scripts. One item.
- "Reduction families" and "Model architecture" both state what is held fixed so that outcome
  differences are attributable to the reduction: the taxonomy covered, the encoder covered.
  One item.
That leaves four items: algorithm, target definition, scale, and what is held fixed.

**Opening paragraph.** The delimitation-versus-limitation sentence is load-bearing and
already short. Keep it. The label-cost clause ("chosen for a label that is expensive to
obtain and sensitive to structure") argues for the target itself, not for the section's
scope, and reads like it belongs earlier in the introduction instead of here.

**Worked rewrite**, the graph-scale item, showing the compression the exemplars use:

Current, three sentences, about 45 words:
> The corpus is bounded at an intermediate rather than industrial scale. RQ1 needs a
> full-graph baseline, and a baseline that does not fit in memory cannot be measured. The
> AIGs of millions to billions of nodes invoked in the motivation are out of reach here.

Compressed, one sentence, about 33 words, reason welded in with a colon instead of argued
across a second sentence:
> The corpus is bounded at an intermediate rather than industrial scale: RQ1's full-graph
> baseline has to fit in memory, so the million- to billion-node AIGs the motivation invokes
> are out of reach here.

Same claim, same reason, same forward-reference slot, one sentence instead of three. Applied
to all six items (four after merging), the section lands close to the 15 to 20 line target
without losing an exclusion or a reason.
