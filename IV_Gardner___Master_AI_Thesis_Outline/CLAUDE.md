# Thesis writing

Applies to all prose in `.tex` files (and to abstracts, captions, and figure text). It does **not** apply to code, code comments, commit messages, or chat, where casual language is fine. The em dash/en dash ban and other universal constraints live in the root [CLAUDE.md](../CLAUDE.md) and apply here too.

**Overriding goal: it must read as human, and specifically as Isabella's writing.** Prose that scans as generated text fails even if it is grammatical and formal. The samples in `~/Documents/MyWork/` are the reference for what "correct" sounds like; when a rule below and the samples disagree, the samples win. The one exception is the flagged-phrase inventory in [Anti-tells for generated prose](#anti-tells-for-generated-prose), which outranks the samples, because the samples are unpublished drafts and contain some of the phrasing that inventory bans.

### Where the thesis lives, and how it reaches the supervisor

Three git locations, and they are **not** connected by submodules. Nothing about this is automatic.

| Location | What it is |
|---|---|
| `~/thesis-main`, branch `main` | The working checkout. Approved, supervisor-facing content. **Write here.** |
| `~/data-gen-rand-abcd`, branch `thesis-outline` | The full working draft, roughly 12,000 lines ahead of `main` in chapters 2 to 7, plus planning notes. Reference and source material. |
| `github.com/bellavg/IV-Gardner-msc-thesis-latex` | The Overleaf mirror the supervisor reads. A flattened snapshot, not a branch of this repo. Shares no history with it. |

`main` and `thesis-outline` drift on purpose. A chapter moves to `main` when it is approved, copied **per file**, never by syncing whole trees: `thesis-outline` still holds older copies of files `main` has since corrected, and a bulk copy silently reverts them.

**Publishing is one command, run from `~/thesis-main`:**

```bash
bash IV_Gardner___Master_AI_Thesis_Outline/sync-overleaf.sh
```

It pushes the current branch to `origin` first, then replaces the mirror's contents with the allowlisted files as of `HEAD`. A failed origin push aborts before the mirror is touched, so Overleaf never shows work that exists nowhere else. It publishes **committed** state; uncommitted edits stay local.

Rules that hold regardless of what is convenient:

- **Never run `sync-overleaf.sh` without being asked.** It reaches the supervisor and pushes to two remotes. Committing is not publishing. Offer, then wait.
- **Never push to the mirror by hand,** and never edit on the Overleaf side expecting it to survive. The script wipes and rewrites the mirror on every run.
- **To publish a new file, add it to the `PUBLISH` allowlist in the script.** The list is private-by-default: a new notes file anywhere under this directory is excluded automatically. Do not work around it by renaming or relocating a file into a published path.
- Only the allowlist reaches the mirror: `msc_thesis.tex`, `mscaithesis.cls`, `README.md`, `sections/**/*.tex`, `bibliographies/**/*.bib`, and `media`. `CLAUDE.md`, every `.md` note, and these scripts exist on `origin` only, which is why the origin push is not optional.
- The script pushes the **whole branch**, not just this directory. On `main` that includes `src/`. Check `git log origin/main..main` before publishing if that matters.

**Pulling Overleaf edits back.** Edits made on the Overleaf side are real work that lives only on the mirror, and the mirror's files sit at its root while this repo nests them under `IV_Gardner___Master_AI_Thesis_Outline/`, so no merge can do it. Diff the mirror against its last sync point and apply with a prefix:

```bash
git fetch git@github.com:bellavg/IV-Gardner-msc-thesis-latex.git main
git log --oneline FETCH_HEAD          # find the newest "Sync thesis source from <rev>" commit
git diff <that-sync-commit> FETCH_HEAD | git apply --directory=IV_Gardner___Master_AI_Thesis_Outline
```

This applies cleanly only if `main` has not moved since that publish. If it has, use `git apply -3` and resolve. Read the diff before applying: hand edits made in Overleaf have introduced grammar errors before.

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

- **Almost no first person.** The agent is the work, not the author: "this work introduces," "this thesis evaluates," "the proposed method achieves." The two conference drafts contain zero instances of *we* or *our*; `aigverse` and the bachelor thesis use them fewer than ten times each. Write the thesis with none.
- Passive voice for method steps ("the graphs were sparsified"), active voice for contributions ("this work introduces").
- Enumerated contributions and scope statements inline: "(i) ..., (ii) ..., and (iii) ...".
- Problem → gap → contribution ordering in every intro-like passage.
- **A named obstacle carries the introduction.** Each paper names its blocking problem as a term ("structural bias", "the scale bottleneck"), defines it in one sentence, and returns to that exact term. The introduction is built around the obstacle, not around the field's importance.
- **Terminology repeats verbatim.** A concept gets one name and keeps it. Renaming it for variety is a generated-prose tell and does not appear in her samples.
- **Magnitudes carry numbers.** "17.2× faster", "$r = 0.72$", "100 % structural validity", "three to four orders of magnitude". A claim about size, speed, or strength with no figure attached is unfinished prose.
- American spelling and the Oxford comma: *modeled*, *materialized*, *optimized*, *centered*.

**Caution: the samples are drafts, not exemplars of every sentence.** `AIG_Generation_ASP_DAC.pdf` in particular still carries unedited phrasing that the anti-tell inventory below bans ("plays a critical role", "sets the stage for", "key properties", a participial tail listing four future applications). Take structure, terminology, person, and density from the samples. Do not lift their phrasing.

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

- Avoid defaulting to *delve, leverage, crucial, pivotal, comprehensive, robust, landscape, foster, showcase, underscore, nuanced, cutting-edge*. Exception: a term that is standard in EDA/ML usage ("robust estimator", "the loss landscape") is fine.
- Also flagged, from the corpus studies cited on the Wikipedia page below: *additionally* (especially sentence-initial), *align with*, *boasts*, *bolstered*, *emphasize*, *enduring*, *enhance*, *garner*, *highlight* (as a verb), *interplay*, *intricate*, *key* (as an adjective), *meticulous*, *tapestry*, *testament*, *valuable*, *vibrant*. These co-occur: where one appears, look for the others.
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

### The motivation section argues, it does not introduce

`sections/1-introduction/1.1-motivation.tex` is the counterpart to the rule above. Concepts that chapter 2 introduces are *motivated* here and introduced there. The motivation answers four questions and stops:

1. **The core problem**, stated broadly, carried by the named obstacle the introduction is built around.
2. **The consequence.** Who is blocked, and at what cost. The "so what".
3. **The gap**, in a sentence or two, at the level of approaches rather than named systems.
4. **The objective**, in outline. What the work sets out to establish, not how.

What does **not** belong:

- **Literature review.** Prior work appears only where it names the specific limitation this thesis fixes. Named systems, their architectures, and the citations behind them belong in related work. If a passage reads as a survey of who did what, it is in the wrong chapter; point to the chapter that does the reviewing instead.
- **Method detail.** No experimental setup, no equations, no algorithm or command names, no configuration, no dataset construction. The approach stays at the level of what is done and why.
- **Sweeping openers.** No history of computing, no "in today's" framing, no appeal to the field's importance in the abstract. Open inside the specific domain.
- **Overblown claims.** No promise the results cannot support, and no assertion that the problem blocks an entire field. State the constraint and the cost it imposes.
- **Detailed results.** One sentence teasing the main finding is allowed at the end. Numbers, tables, and conclusions are not.

### Anti-tells for generated prose

**Definitive reference: <https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing>.** That page is an evidence-backed catalogue of the patterns that mark text as machine-written, and it is the specification for what this thesis must not sound like. Consult it directly when editing a passage that feels generated but you cannot say why. Its "Content", "Language and grammar", and "Style" sections are the relevant ones; the markup, citation-bug, and Wikipedia-process sections are not.

The patterns below are the subset that actually bites in an EDA thesis. They make a paragraph read as machine-written even when every sentence is defensible.

1. **Uniform sentence length.** Generated text sits at 20 to 25 words per sentence for a whole paragraph. Her samples mix an 8-word sentence against a 45-word one. Vary deliberately.
2. **Participial tails.** An "-ing" clause bolted onto a sentence to editorialise about what precedes it: "..., highlighting the importance of structural fidelity", "..., reflecting a broader shift toward learned heuristics". This is the most frequent single tell. Cut the clause. If the observation carries weight, it earns its own sentence with evidence behind it.
3. **Copula avoidance.** *serves as, functions as, represents, stands as, boasts, features, offers* where *is* or *has* would do. Write "the AIG is the input representation", not "the AIG serves as the input representation". Corpus studies measured a 10 % drop in *is*/*are* in academic writing after 2023; plain copulas now read as a human signal.
4. **Negative parallelism.** "not just X, but Y", "not X, but rather Y", "X rather than Y" used as a rhetorical frame rather than a real contrast. State the claim positively.
5. **Reflexive tricolon.** Three parallel items whenever a list appears. Use two items, or four, when that is what the content has.
6. **Elegant variation.** Renaming one thing across a paragraph (graph, structure, representation, topology) to avoid repeating a word. Repeat the word.
7. **Significance inflation.** "plays a critical role in", "is a testament to", "reflects broader trends in", "setting the stage for", "marks a shift toward", "the evolving landscape of". The reader already accepts that logic synthesis matters. Argue the specific claim instead.
8. **Structure announcements.** "This section proceeds in three parts." Cut it. The headings already say this. The "remainder of this paper is structured as follows" roadmap is the one sanctioned exception, once, at the end of the introduction, as in her samples.
9. **Summarizing morals at paragraph end.** "This underscores the importance of careful evaluation." Delete the sentence; it adds no claim.
10. **Symmetric hedging.** "While X offers advantages, it also presents challenges." Name the actual limitation and its cost, or say nothing.
11. **"Despite these challenges" closers.** A passage that concedes difficulties and then reasserts promise in the same breath. State the limitation and stop.
12. **Vague attribution.** "researchers argue", "studies have shown", "it is widely recognised", or "several works" in front of one citation. Name who, with a reference, or cut the sentence.
13. **Title Case headings**, curly quotation marks, and bold used for emphasis inside prose. Sentence case, LaTeX quoting (`` `` `` and `` '' ``), and `\emph{}` used sparingly.

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
