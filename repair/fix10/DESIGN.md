<!-- Provenance (TriPhysBench release): fix_code/evaluation/fix10/DESIGN.md Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos). Release edits only: API keys, endpoints, model ids and machine paths come from environment variables (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical. -->
# fix10 — repair of the problem QAs (design 10/11, criteria v2 10/12)

## Criteria v2 (user, 10/12) — these replace the counting rules below
| Label | v2 decision |
|---|---|
| `W_query_added` | Not a problem: the solver also sees the video. |
| `W_base_conflict` | Re-judged **with video frames**. The 10/11 VALID judge saw only the question text. `rejudge.py` shows deepseek-flash the frames plus the "objects seen" list. A conflict counts only if the verdict is "alternative" or "both". |
| `M_both_valid` | Not a problem: several solution methods are natural. |
| `M_alt_more_standard` | Fixed. The GT is recomputed with the standard model. Add one assumption sentence only if no default convention exists; it states a physical condition only, with no formula, method or number. |
| `V_invented` | A typical / standard value is fine when the base query gives a unique value. Only an **identity** the frames do not show counts (NBA player, Mikan drill, professional, deep-groove bearing, mechanical advantage of 2, …). Fix it with neutral wording. |
| `V_contradicted` / `V_absent` | Counted after the dense re-check (`recheck_ns.py`, ≤48 frames, ≥ every 0.2 s), in both modes: notshown and contra. |
| `W_base_unreach` / `W_no_base_query` / `S_*` / `C_inconsistent` | Counted as before. |

**Repair approach: "小改为主、重写兜底".** `fix10.py` runs `EDIT_ROUNDS = 2` small-edit rounds, then whole-rewrite rounds as the fallback.
- **Small edit.** A repair plan, made with the frames, assigns each web constant one action:
  - keep;
  - requery: a better base query for the same quantity;
  - qualify: name the object or material as the frames show it;
  - replace: another property of the same object.

  Re-sourced constants are verified with Serper (unique value) **before** the edit is applied. Only the offending phrases are rewritten; every other sentence is copied.
- **Acceptance after any edit or rewrite** (user: "修改后的核验也要做，要用 Serper"):
  - `recount.check` on the new QA: a Serper base query for **every** constant (reach + values substituted into the code with the official ACC), a fresh M check, video frames, static checks and C;
  - conflicts re-judged with video;
  - v2 labels;
  - fix9 static checks and fidelity review.
- **Pipeline:** `recount_final.json` → `rejudge.py` → `recount_v2.json` (`labels2` / `needs_fix2`) → `fix10.py`.
- **Type and difficulty design are hard checks in acceptance** (user 10/12, `form_problems`; the pilots showed drift: one rewrite changed Statics into heat capacity and went from 1 video reading to 3). Kept and checked:
  - `question_type` (copied from the old QA);
  - #web constants, #hidden and #video readings;
  - the physics area (first part of `physics_category`; a small edit keeps the old label);
  - the answer form: TYPE_C starts with Yes / No; TYPE_D states a comparison; the TYPE_B range comes from a constant whose standard is a range.

  443 of the 445 current QAs pass the form rules.
- **Meaning of #hidden (user 10/12):**
  - A: the number of CV values hidden. Every web-search value is always hidden. A rewrite keeps `hidden_cv_chosen`, the video readings and the readings the old question states (checked).
  - B: the number of hidden web-search values.
- **Making a non-unique constant unique (user 10/12, v6):**
  - The question states one explicit assumption that names a material, grade, standard or condition for the same object, e.g. "Assume the marble is soda-lime glass." or "Assume the ball is a FIBA size 7 basketball.". The base query uses its words.
  - The assumption must be plausible for the frames and contain no number, formula or method.
  - Asserting such an identity as a scene fact is still V_invented.
  - The edit plan gives 3-5 (assumption, query) candidates per constant. They are Serper-verified in parallel and the first unique one is used.
  - The object and the asked quantity never change. The reviewer's `drift` criterion rejects rewrites that move to another object or quantity.
- **Round schedule (user 10/12):** small edits continue while the failures are only wording or review problems. The QA switches to a whole rewrite, for its remaining rounds, only when a constant has no unique value: either the re-sourced constant fails Serper verification, or the Serper re-check of the edited QA still finds a W problem. 5 rounds in total.
- **QAs still unresolved after all rounds are deleted from version 08** (user 10/12). Their count is reported by split, class, difficulty and type.

## v7: query-first repair (user 10/12, replaces the v6 rewrite fallback)
Most problems are a base query that does not return one value; a range counts as one value if it is unique. So a web constant is repaired through its query, and the question text changes as little as possible. Of the 255 needs-fix QAs, 168 have a W label and 112 have only a W label.

**Candidates.** Per problem constant, the edit plan (deepseek-flash, with frames) gives 3-5 base-query candidates, verified in parallel:
- `requery`: only the query changes; every word is in the question or shown by the frames;
- `qualify`: the query names a material, grade, standard or condition, and one explicit "Assume …" sentence added to the question names it (plausible for the frames; no number, formula or method).

Each candidate is resolved in this order:
1. **The old value is returned uniquely** → only the query and the assumption change; the answer stays the same.
2. **One different value is returned uniquely** → the query and the value change. The code, answer and answer_range are recomputed. For B, `hidden_cv_chosen` follows the new value.
3. **`replace`** (another property of the same object) → only if no query for the same quantity works.
4. **Nothing works** after 5 rounds → the QA is deleted.

"Typical X" quantities are fine when their query returns one value. There is no whole rewrite.

**`unique()` — one test, used by the pre-check and by acceptance.** Earlier the pre-check (12 % / 20 % width) and acceptance (ACC) used different rules. Now both use `recount.base_const`:
1. **Extraction:** the Serper top-10 text the solver sees is extracted twice. A second, object-tagged pass labels each value with what it refers to. The union is kept, so a missed value cannot make a constant look unique, and values for another object/size/grade are dropped.
2. **ACC test:** every value is substituted into the code and scored with the official ACC against the reference answer. A value with ACC < .5 is a conflict unless the question + frames single out the reference (video re-judge, as in `rejudge.py`). For TYPE_A, ACC ≥ .5 ⇔ an answer change of ≤ ~10.7 %.
3. **Support:** at least 2 different sites state the agreeing value, or the quote is a standard / regulation / handbook value.
4. **Stability:** a paraphrase of the query (same qualifiers, other word order or synonyms) must not return a conflicting value.

**Range ↔ single value.** Either may replace the other if unique.
- TYPE_A ↔ TYPE_B follows the final answer: a range answer needs a web constant that is a range, and then the type is TYPE_B; otherwise TYPE_A.
- TYPE_C / TYPE_D keep their type.
- Difficulty counts never change. Type changes are counted and reported.

**Other labels.** V / S / M problems are fixed by small edits of the offending phrases only. M_alt: the code uses the standard model.

**Review.** gpt-4o reviews on the ACC-judge endpoint, with the same frames sent as images (tested 10/12). The writer and the reviewer are now different models. It gets the old answer and the list of changes, and checks two new criteria:
- `added_physics`: a coefficient, multiplier, efficiency or effect that neither the old question nor the frames support;
- `answer_change`: the answer changes for a reason other than the listed constant or modelling changes.

If the review fails to return, that counts as a problem, not a pass.

**Other fix.** fix9 required C/D to keep answer_range; that rule is dropped (only TYPE_B needs answer_range).

## Agreed criteria (user, 10/11)
- **Web constant**:
  - The annotator's base query must be writable from the question. Every query term must be stated or implied by the question, or be generic wording.
  - That query must return the reference value, and every value of the quantity in its results must agree with it. "Agree" means the official ACC is ≥ .5 after substitution into the code.
  - Values that only other queries find do not count.
- **Modelling**:
  - Every interpretation choice that changes the answer must be fixed by the question: a stated assumption or convention, never a method hint.
  - If the alternative is more standard than the reference, recompute the GT with the standard choice and state it.
- **Video**: every statement about the scene must be visible or plausible in the video. Nothing contradicted, and no invented brand, model, grade or standard. deepseek-flash takes image input (not `video_url`), so frames are sampled from the full eval video.
- **Never**:
  - widen the answer range or accept several answers;
  - put web or hidden values in the question, or mention tools or searching;
  - assign one object another object's property.
- **Keep per QA**:
  - class, split and video;
  - `question_type`;
  - the counts (#video readings, #hidden, #web constants), which fix the difficulty exactly (A: #hidden = 0/1/2; B: #ws = 1/2/3);
  - for A, the video readings and `hidden_cv_chosen` (CV values, not changed).

## Step 1 — recount (`recount.py` → `recount.json`)
Labels per QA:
| Label | Meaning |
|---|---|
| `W_base_unreach` | The base query does not return the reference value. |
| `W_base_conflict` | The base results hold another valid value that makes the answer wrong. |
| `W_query_added` | The base query has a term the question lacks, and the question alone does not reach a unique value (10/11 question-only queries). |
| `M_both_valid` / `M_alt_more_standard` | From the 10/11 M step; rerun if the question has changed since. |
| `V_contradicted` / `V_invented` | The question describes the scene wrongly. `V_not_shown` (frame-sampling gaps) is reported but not counted. |
| `S_tool_wording` / `S_hidden_leak` / `S_web_leak` | Static checks on the question text. |
| `C_inconsistent` | The code does not reproduce the answer. |

`needs_fix` = any label except `V_not_shown` / `V_error`.

## Step 2 — rewrite (`fix10.py`, deepseek-flash, per needs_fix QA)
Context given to the writer:
- the original QA and its recount evidence (conflicting quotes, model alternatives, video claims);
- the video description: stage1_omni_info, stage2 event graph, stage25 CV physics (A_class_Benchmark/output, B_class_Benchmark/outputs), plus the recount "objects seen" list;
- the other QAs of the same video, so the new one does not duplicate them.

Constant-first generation (the key change vs L2 / fix9: verify uniqueness before writing the question):
1. **Plan.** Choose the physics (same `physics_category` where possible) and exactly n_ws web constants. Each must be a property of an object or material that the video shows or plausibly implies:
   - the material of a visible object (steel, water, rubber, wood);
   - a regulation or standard object identified by the setting (an adult basketball game → size-7 ball);
   - a physical constant of the scene.

   Generic "typical X" properties of unidentified objects (a wheel, an eraser, a person's mass) are not allowed. Such quantities become given inputs or video readings. For each constant, give a base query written only with words the question will contain.
2. **Verify the constants (before writing the question).** Run the Serper base query. Extract every value of the quantity with a verbatim quote. The constant must be unique: all values agree within 2%, or, for a TYPE_B constant, the standard range itself. Otherwise go back to 1 with the conflicting quotes, at most 3 times.
3. **Write.** Produce the question, annotated_solution, sources (verified quote, url, query) and code. The code is run, and the answer and answer_range come from its output (3–4 significant figures). B `hidden_cv_chosen` = the new web values (same count). A keeps its CV readings.
4. **Accept.** All of these must pass:
   - recount W / M / V / S / C on the new QA (M: fresh MODEL_P + MODEL_V);
   - type, difficulty counts, no values in the question;
   - a fidelity check that the scene is the same video and no other object's property is used.

   On failure, go back to 1–3 with the problems, at most 3 rounds. Still failing → `unresolved` (reported, not written).

## Step 3 — apply and re-evaluate
- `fix10_apply.py` writes the accepted QAs into the 6 files (backups), creates `question_versions/08_fix10_*`, uploads to data/v9 and checks md5.
- Test: re-run only the rewritten test QAs for the 6 zero-shot models and the key trained models (unified eval); report A / B separately on the full test set and on the unchanged subset (paired CI).
- Train: rebuild the SFT data from the new train set before any new training.
