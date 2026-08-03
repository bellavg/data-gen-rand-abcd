# Thesis writing

Applies to all prose in `.tex` files (and to abstracts, captions, and figure text). It does **not** apply to code, code comments, commit messages, or chat, where casual language is fine. The em dash/en dash ban and other universal constraints live in the root [CLAUDE.md](../CLAUDE.md) and apply here too.

**Overriding goal: it must read as human, and specifically as Isabella's writing.** Prose that scans as generated text fails even if it is grammatical and formal. The samples in `~/Documents/MyWork/` are the reference for what "correct" sounds like; when a rule below and the samples disagree, the samples win.

### Register

Publication-ready scientific English, at the level of a submitted conference paper. Every sentence should be one you would submit to a venue without further editing.

- No contractions, colloquialisms, or conversational asides.
- No filler hedges ("it is worth noting," "arguably," "quite"). Hedge only where the evidence genuinely is uncertain, and hedge precisely.
- Define each technical term on first use with its abbreviation in parentheses: "Logic Synthesis (LS)". Use the abbreviation thereafter.
- No unexplained pronouns ("this," "it") without a clear antecedent.

### Voice: match the author's own samples

`~/Documents/MyWork/` holds the author's prior work. **Read at least one relevant sample before drafting a new section**, and match its cadence rather than inventing a register:

| File | Use it for |
|---|---|
| `AIG_Generation_ASP_DAC.pdf` | Closest genre match: AIG/EDA conference paper. Abstract, intro, contributions. |
| `AIG_Similarity_Metrics_DATE__Copy_.pdf` | Metrics/evaluation framing, results prose. |
| `BachelorThesis_IVGardner.pdf` | Thesis-length structure, related work, discussion. |
| `Psychedelics Essay.txt`, `AW Fin. Assignment Isabella Gardner.txt` | Argumentative/survey prose, connective style. |

Recurring traits in those samples, to preserve:

- **Almost no first person.** The agent is the work, not the author: "this work introduces," "this thesis evaluates," "the proposed method achieves." Across all three papers, first-person pronouns appear fewer than five times total.
- Passive voice for method steps ("the graphs were sparsified"), active voice for contributions ("this work introduces").
- Enumerated contributions and scope statements inline: "(i) ..., (ii) ..., and (iii) ...".
- Problem → gap → contribution ordering in every intro-like passage.

### Conciseness (the document is still an outline)

The outline stage rewards density, not volume. See the [page limit note in the thesis README](README.md).

- Write the shortest sentence that carries the full claim. If a paragraph can lose 30% and lose no content, cut it.
- One idea per sentence. Prefer a period over a semicolon.
- Delete throat-clearing openers ("In this section, we will...", "It is important to note that..."). Open with the claim.
- Do not pad an outline stub into paragraphs to make it look finished. A precise two-sentence placeholder beats a vague page.

### Comments in the `.tex` source

**The source is handed in with the document, and a comment that reasons with itself reads as a private conversation rather than an author's note.** Default to no comment. Reasoning that has to survive belongs in the repository, not in the file being submitted.

- One line, no rationale. `% TODO: measure before defending this` is fine. A multi-line note explaining why a passage sits in this chapter rather than another is not.
- No bullet lists, no `\ref` chains, and no second-person instructions ("keep it to one sentence each", "do not report it as tested") inside a comment. That register is the tell.
- Do not record why a decision was made. Either the prose carries the reason or the reason does not need writing down.
- Outline scaffolding, a heading whose intended content is written as comments, is the one exception, and only while that section is still a stub. It goes before the section is shown to anyone.

**Exception, currently active: preserved prose from condensation.** While the document is at the outline stage, prose removed by a condensation pass is kept in place as a `%` comment, immediately below its replacement and introduced by a single `% Original, condensed above:` line. Roughly 21,700 words are preserved this way. **Do not sweep, delete, reflow or un-comment those blocks.** They are the full text the author will use in the real thesis, and deleting them destroys work that cannot be regenerated. This exception overrides the general rule that commented-out prose is restored or deleted, and it stays in force until she says the condensation is final.

Sweep the *other* comments before sending anything out, and read what you are deleting first: `grep -rn "^\s*%" sections/`. A `% TODO` line and a `% Original, condensed above:` block are both deliberate; neither is sweepable.

### Word and punctuation discipline

Follow the flagged-term and punctuation rules in `~/.claude/plugins/cache/academic-research-skills/academic-research-skills/*/academic-paper/references/writing_quality_check.md`. The ones that bite most often:

- Avoid defaulting to *delve, leverage, crucial, pivotal, comprehensive, robust, landscape, foster, showcase, underscore, nuanced, cutting-edge*. Exception: a term that is standard in EDA/ML usage ("robust estimator") is fine.
- Semicolons: ≤ 2 per 1000 words.
- Do not open consecutive paragraphs with a colon-plus-list.

### Mathematical language first

**Where a claim can be stated as a definition, an equation, or an inequality, state it that way.** Formal notation is the register this field reads in, and it is also the shortest form of the claim, which is what the outline stage rewards. Prose then reads the formula out and says why that quantity is the right one; it does not paraphrase the formula in words.

- Every metric, target, ratio, and operator gets a defining equation the first time it is reported, in a numbered and labelled environment, and is referred to by number afterwards.
- Reuse the notation already fixed in the problem formalization (`sec:prelim:formalization`) and in the architecture equations. Do not introduce a second symbol for a quantity that already has one.
- Sets, quantifiers, cardinalities, and thresholds belong in math mode, not in words: $|\mathcal{V}(R(G))| / |\mathcal{V}(G)|$, not "the ratio of kept nodes to original nodes".
- State a threshold or a comparison as the relation it is ("a reduction pays for itself once $n > c_R / s$"), not as a sentence describing that a relation exists.
- Do not add ceremonial math. If a sentence is already exact and no reader would need the symbols to reproduce it, leave it prose. When genuinely unsure, add the equation.

### Citations come from DBLP

**Every entry in `bibliographies/references.bib` must be checked against DBLP before it is cited, and rechecked whenever a citation is touched.** DBLP is the authority for authors, venue, year, and publication status; guessing any of those, or leaving a placeholder in a field, is a citation error.

- Search: `curl -s 'https://dblp.org/search/publ/api?q=<query>&format=json&h=10'`. Fetch one record's BibTeX: `curl -s 'https://dblp.org/rec/<dblp-key>.bib?param=1'`.
- Compare field by field against the local entry and fix the local entry to match: full author given names (not initials, never `TODO`), the real venue, and the correct entry type.
- Prefer the published record over the preprint when DBLP lists both. Cite the arXiv version only when there is no published record.
- If DBLP has no record at all (common for a preprint from the last few months), keep the arXiv entry and say in a `note` that it is unindexed. Do not fabricate the missing metadata.
- Verifying the arXiv landing page directly needs `curl` plus `pdftotext`; the fetch tool is blocked on that host.

### Level of abstraction (supervisor feedback)

Direct feedback received: the writing is **too code-specific and hardware-specific for a paper**. Raise the level of abstraction. Prose describes concepts and mechanisms; identifiers and machine details belong in the experimental setup, the reproducibility subsection, tables, or the appendix.

- **Name the concept, not the identifier.** Write "colour refinement", "maximal fanout-free cone contraction", "convolution matching", not `wl`, `mffc`, `convmatch`. Introduce the concept name first, and give the implementation name once, in the setup section, if a reader needs it to reproduce.
- **No file names, function names, flags, config keys, or class names in prose.** No `config.VALID_ALGORITHMS`, no `--split_by`, no script paths.
- **Generalise hardware.** "Peak memory", not "VRAM"; "exceeds device memory", not "OOM crash"; "modest hardware", not "a single H100". Specific devices and library versions are stated once, in reproducibility, and never again.
- **Describe the mechanism, not the call sequence.** "Edges are removed in inverse proportion to their span" beats "the sparsifier iterates the edge list and drops entries below the threshold."
- Exception: the experimental setup and reproducibility subsections exist precisely to be concrete. Do not abstract away the numbers there.

### Preliminaries and related work introduce concepts, nothing else

**The preliminaries and the related work introduce concepts. They do not mention this thesis.** Author feedback, given repeatedly and emphatically: a preliminaries section that explains how the work applies a concept has put method content in the wrong chapter.

Define each concept so a reader can understand what it is, why it matters in the field, and enough about it to follow its later use. Then stop. Judge every sentence by "would this sentence exist in a textbook chapter on this topic?" If it only exists because of this thesis, it belongs in the methodology, the research questions, or the discussion.

Specifically, these do **not** belong in preliminaries or related work:

- How a concept is applied in this work, or which of its variants this work chose.
- Feature representation: tensor conversion, one-hot encoding, input schemas, feature dimensions.
- Motivation for this thesis, problem statements, gaps framed as this work's opportunity.
- Cross-references forward into the methodology that argue rather than point.
- Naming the target algorithm, the corpus, the splits, or the reductions as *this work's* choices.
- Mentioning AIGs inside the GNN preliminaries, or the ML pipeline inside the AIG preliminaries. Keep the two vocabularies separate until the methodology brings them together.

The inverse failure also counts: a concept the later chapters lean on must be introduced here **well enough to carry that weight**. If the methodology assumes the reader knows how a synthesis algorithm behaves, or what reconvergence costs, the preliminaries owe that explanation, framed as the concept rather than as this work's use of it.

### Anti-tells for generated prose

These are the patterns that make a paragraph read as machine-written even when every sentence is defensible:

1. **Uniform sentence length.** Generated text sits at 20 to 25 words per sentence for a whole paragraph. Her samples mix an 8-word sentence against a 45-word one. Vary deliberately.
2. **Reflexive tricolon.** Three parallel items whenever a list appears, and the "not merely X, but Y" construction. Use two items, or four, when that is what the content has.
3. **Structure announcements.** "This section proceeds in three parts." Cut it. The headings already say this.
4. **Summarizing morals at paragraph end.** "This underscores the importance of careful evaluation." Delete the sentence; it adds no claim.
5. **Symmetric hedging.** "While X offers advantages, it also presents challenges." Name the actual limitation and its cost, or say nothing.

### Using the academic-research-skills package

**Read its reference files as a ruleset; do not run its generative modes on this thesis.** The content is hers. `/ars-full`, `/ars-revision`, and the `academic-paper` skill in full mode draft and rewrite wholesale, which replaces her argument with a generated one. That is the wrong instrument for a document that already has an author.

Reference files (read directly, apply as an editing checklist) live in:
`~/.claude/plugins/cache/academic-research-skills/academic-research-skills/*/academic-paper/references/`

| File | Read it when |
|---|---|
| `writing_quality_check.md` | Any prose edit. Flagged terms, punctuation limits, throat-clearing, sentence-length variation. |
| `academic_writing_style.md` | Setting register. The Engineering/CS block is the right one here. |
| `writing_judgment_framework.md` | Deciding what to cut. The clarity test and load-bearing vs supporting paragraphs. |
| `abstract_writing_guide.md` | Writing or revising the abstract (200-word limit, no citations). |
| `intro_title_rhetoric_guide.md` | Reworking the introduction's problem/gap/contribution move. |
| `paper_structure_patterns.md` | Deciding what belongs in which chapter. |
| `statistical_visualization_standards.md` | Figure captions and results plots. Captions must be readable without the body text. |

Commands that are safe to run, because they report rather than rewrite:

- `/ars-citation-check` for a citation error report against `bibliographies/references.bib`. Pairs with the DBLP rule above.
- `/ars-reviewer` for a simulated peer-review panel, once a section reads as a full draft rather than an outline. Advisory only; apply its findings by hand.
- `/ars-outline` for a stub section that needs structure before prose exists. It produces an outline and evidence map, not a draft.
- `/ars-3w` for a WHY/HOW/WHAT scan when triaging papers for related work.
- `/ars-disclosure` if the university requires an AI-usage statement.
