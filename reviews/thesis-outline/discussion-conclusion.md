# Outline review: Discussion, Conclusion, Appendix, front matter

Unit: `sections/5-discussion.tex`, `sections/6-conclusion.tex`, `sections/7-appendix.tex`,
`sections/0-list-symbols.tex`, `msc_thesis.tex`.
Baseline: snapshot commit 353c4dd. Review branch: `review/discussion-conclusion`, commit a3bb86d.
Build gate held throughout: 0 errors, 0 undefined references, 0 undefined citations, 105 pages.
chktex on all five files: clean before and clean after.

## Template-compliance checklist

| Requirement | Result |
| --- | --- |
| Discussion opens with comparison to the state of the art | Pass. `sec:discussion:relatedwork` is the first section of the chapter. |
| Limitations cover reproducibility | Pass. Written out, not a stub: single seed, unpinned tool version, partially regenerable corpus. |
| Limitations cover scalability | Pass. Written out, and states explicitly what does and does not extrapolate. |
| Limitations cover generalizability | Pass. Three axes held fixed, each named. |
| Limitations cover reliability and validity | Pass. Construct and internal validity both treated, with the RQ1a measurement flagged as pending. |
| Ethics discussed | Pass. Four angles: compute cost, benchmark provenance and licensing, accessibility, dual use. |
| Each research question answered in the Conclusion | Pass for RQ1 to RQ5 as heading stubs. RQ1a was missing and is now slotted under RQ1. |
| Each answer paired with its qualifying limitation | Pass. The mapping in the writing plan is now recorded in full in the source, RQ1a included. |
| Appendix holds what the main text points at | Partial. Three of five appendix sections are referenced from the main text; two are not. See finding A1. |
| Thesis readable without the appendix | Pass. Nothing in the main text defers a claim to the appendix; the pointers are all to reproduction detail. |
| Abstract skeleton covers problem, gap, method, headline result, transfer finding | Pass. Six planned sentences, one per element, with the two result slots left as placeholders. |
| Abstract constraints recorded (200 words, no citations) | Fixed. The constraints were absent from the skeleton and are now written next to it. |
| List of symbols matches the notation in use | Fixed at stub level. The file was still the template example; it now carries the actual inventory as TODO lines. |

## Findings, ranked

### Fixed on this branch

1. **RQ1a had no answer slot in the Conclusion.** RQ1a is a full research question in the
   introduction, it has its own results subsection, and the Discussion's validity subsection
   defers to it, but the Conclusion had no place to answer it and the limitation-pairing map
   in the source skipped it. Added a one-line TODO under RQ1 and extended the pairing comment
   to cover RQ1a against the validity limitation.
2. **The list of symbols was untouched template boilerplate.** The thesis fixes roughly
   twenty-five recurring symbols across the preliminaries and methodology chapters, and the
   front-matter stub recorded none of them. Added five one-line TODO stubs grouping the
   notation actually in use (AIG objects, structural quantities, task objects, reduction
   operators, encoder state) plus one line for the abbreviations in use. The chapter remains
   commented out, so nothing new renders and the page count is unchanged.
3. **Two comparison subsections were bare headings.** `Positioning Against Graph Reduction Work`
   and `Where Direct Comparison Is Not Possible` carried no intent stub, which put them below
   the standard every other empty subsection in these chapters meets. Added a one-line stub to
   each, naming what the subsection has to establish.
4. **Future work did not link back to the scope decisions it reopens.** The scope subsections
   point forward to future work, but the loop closed in one direction only. Added a one-line
   backlink to each of the four future-work subsections, against the algorithm, reductions,
   scale and prediction scope items respectively.
5. **The appendix chapter was still titled "First Appendix".** Retitled to
   "Experimental Details and Extended Results", which is what its five sections hold. The label
   is unchanged, so the one main-text reference to it still resolves.
6. **The abstract skeleton did not record its own constraints.** The word cap and the
   no-citations rule are the two things easiest to violate when the abstract is written last.
   Added as one line above the skeleton.
7. **The abstract skeleton used dash punctuation.** Two double-hyphens in the planned sentence
   four. Rewritten with commas.
8. **Ethics sat between the Limitations and the Outlook.** The Outlook stub exists to hand off
   to the Conclusion's future work, so it should be the last thing in the chapter. Moved Ethics
   ahead of it. Section order within the chapter is the only change; no content moved.

### Needs the author

A1. **Two appendix sections are orphaned.** `sec:apx:params` (reduction method parameters) and
`sec:apx:results` (extended results) are not referenced from anywhere in the main text. The
params section motivates itself by the matched-compression confound that RQ4 depends on, so
the natural pointer is from the methodology's reduction sections or from the RQ4 results
subsection. Extended results should be pointed at from the results chapter. Both pointers fall
in other reviewers' files, so they were not added here.

A2. **The methodology chapter points at the appendix chapter, not at the section.**
`sections/methodology/data.tex` line 192 refers a reader to `sec:apx:first_appendix` for command
templates and per-algorithm parameters, which are in `sec:apx:algorithms`. Two other places in
the same chapter already point at the section label correctly. Outside this unit's files.

A3. **The tool version is still unpinned.** Flagged in the appendix and in the reproducibility
limitation. The labels depend on it. Blocked on the author; not resolved here.

A4. **The compute and energy figures are still qualitative.** The ethics paragraph describes the
cost rather than quoting it, and its own TODO says so. Blocked on measurement.

A5. **The RQ1a inflation factor is still a hedge.** The validity subsection carries a TODO to
replace the hedge with the measured factor and to state whether per-design degradation was
uniform or concentrated. Blocked on chapter four.

A6. **Three Discussion sections remain stubs.** Practical Implications, Outlook, and the
interpretation subsections are one-line intents only. That is appropriate at outline stage and
consistent with the writing plan, so it is recorded rather than flagged as a defect.

A7. **The title and the student number are unresolved.** Both carry TODOs in the front matter.
The title TODO is well posed: it names three candidate framings and ties the choice to whether
RQ4 lands decisively. Left as is, since picking a title is the author's call.

## What the ARS passes surfaced

The reviewer skill was run in full mode, framed as an outline-stage content review with sentence
style out of scope. Findings by seat:

- **Editor, template fit.** Confirmed the template requirements individually: comparison first,
  five limitation categories present, ethics present. Caught the missing RQ1a answer slot and the
  Ethics/Outlook ordering, which are findings 1 and 8.
- **Methodology reviewer.** Confirmed the reproducibility subsection is internally consistent
  with the two appendix TODOs on the unrecorded tool version and the unrecorded stochastic seed,
  and that both are correctly flagged rather than fabricated. Raised the orphaned appendix
  sections, which is A1.
- **Domain reviewer.** Flagged the two bare comparison subsections as below the outline's own
  standard, which is finding 3, and the chapter-level appendix reference, which is A2.
- **Cross-perspective reviewer.** Judged the ethics breadth adequate. Raised the missing abstract
  constraints and the untouched symbol list, findings 6 and 2.
- **Devil's advocate.** No critical contradiction found between the Discussion's written claims
  and the writing plan. Two hedges noted and judged sound: the scalability extrapolation claims
  are argued from algorithm properties rather than measured, which the text itself says, and the
  FRAIG negative control is reported as not run rather than glossed. Raised that future work did
  not name the scope decisions it reopens, which is finding 4.

Pass two re-reviewed the edited state against the same checklist and surfaced no new content
findings, so the iteration stopped there.

## Summary for the supervisor discussion

The Discussion and Conclusion meet the template's structural requirements: comparison to prior
work opens the chapter, all five limitation categories are covered, and ethics is discussed
across four angles.
Three of the five limitation subsections are written out in full and are unusually candid; the
scalability one states plainly that the evaluation does not reach the scale the motivation
argues about.
Every research question now has an answer slot paired with the specific limitation that
qualifies it, RQ1a included, which was the one gap in the Conclusion's coverage.
The appendix holds reproduction detail only and the thesis reads without it, but two of its five
sections are not yet pointed at from the main text.
What remains open is measurement rather than structure: the protocol inflation factor, the
compute figures, and the synthesis tool version are all flagged in the source and none were
filled in.
