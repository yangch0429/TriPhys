# Provenance (TriPhysBench release): fix_code/evaluation/scripts/fix8_apply.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix8 (10/11): the one annotation_error of the well-posedness check (evaluation/wellposed, step 1).
  A|test|4|0 jump shot 6.005 m/s, 52 deg, 2.2 m from the three-point line: the ball lands ~4.8 m away, so the old GT
              "height at the three-point distance = -5.51 m" extrapolated the parabola below the floor (conclusion "No"
              right, numbers meaningless); "three-point line" was also open (FIBA 6.75 m arc / 6.60 m corner, NBA 7.24 m)
              -> top of the FIBA arc; ask for the horizontal distance at which the descending ball is back at rim
              height (2.68 m) and compare with 6.75 m (4.07 m short); apex 3.34 m.
              TYPE_B -> TYPE_A: TYPE_B is scored only against a range (metrics_final._acc_type_b; a single-value GT gives 0
              for any answer, so correct "2.68 m, No" answers of GPT-5.5 / Gemini scored 0 in the first try); the asked
              quantity is one distance, so TYPE_A with the number first in the answer. Difficulty unchanged.
Applied to aclass_qa_test.json and aclass_qa_dataset_cleaned.json (backups *.before_fix8.json); log evaluation/fix8_log.json.
Usage: python fix8_apply.py [--dry]"""
import os, sys, json, copy, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix7_apply import FIX, src, code_src, full_qas  # noqa: E402

DRY = '--dry' in sys.argv
CODE = """import math
v0 = 6.005; theta = math.radians(52); h0 = 2.2; g = 9.81; hoop_h = 3.05; R = 6.75
v0x = v0 * math.cos(theta)
v0y = v0 * math.sin(theta)
y_max = h0 + v0y**2 / (2 * g)
t_rim = (v0y + math.sqrt(v0y**2 - 2 * g * (hoop_h - h0))) / g
x_rim = v0x * t_rim
print(f'v0x = {v0x:.3f} m/s, v0y = {v0y:.3f} m/s')
print(f'maximum height = {y_max:.2f} m')
print(f'back at rim height after t = {t_rim:.3f} s, horizontal distance = {x_rim:.2f} m')
print(f'shortfall vs FIBA three-point distance {R} m: {R - x_rim:.2f} m')
print(f'Final answer: No; the ball is back at rim height {x_rim:.2f} m from release, {R - x_rim:.2f} m short of the hoop')"""


def a4_0(x):
    x['question'] = (
        'In the video, a basketball player executes a jump shot with the ball released at a speed of 6.005 m/s. Assuming the '
        'player releases the ball at a typical optimal angle of 52° from a height of 2.2 m, and the shot is taken from the top '
        'of the FIBA three-point arc (straight in front of the basket), at what horizontal distance from the release point '
        'does the descending ball fall back to the 3.05 m rim height? Compare it with the FIBA three-point distance to '
        'predict whether the ball can reach the hoop.')
    x['question_type'] = 'TYPE_A'
    st = x['annotated_solution']
    st[3] = ("Step 4: Use web_search(query='FIBA three-point line distance meters') to retrieve the FIBA three-point distance "
             "at the top of the arc = 6.75 m from the centre of the basket.")
    st[5] = ('Step 6: Use code_executor to compute the projectile motion: v₀ = 6.005 m/s, θ = 52°, h₀ = 2.2 m, g = 9.81 m/s². '
             'v₀ₓ = 3.697 m/s, v₀ᵧ = 4.732 m/s; maximum height = h₀ + v₀ᵧ²/(2g) = 3.34 m (above the rim). The descending ball is '
             'back at rim height 3.05 m when h₀ + v₀ᵧt − ½gt² = 3.05 m, i.e. t = 0.726 s, at horizontal distance '
             'x = v₀ₓt = 2.68 m from the release point.')
    st[6] = ('Step 7: Compare with the FIBA three-point distance: 2.68 m < 6.75 m, so the ball falls below rim height 4.07 m '
             'before reaching the hoop; it cannot reach the hoop.')
    s = src(x, 'web_search', 1)
    s['key_information'] = 'FIBA three-point distance (top of the arc) = 6.75 m'
    out = code_src(x, CODE)
    x['answer'] = ('2.68 m — the ball peaks at 3.34 m and falls back to the 3.05 m rim height 2.68 m from the release point, '
                   '4.07 m short of the 6.75 m FIBA three-point distance, so it cannot reach the hoop.')
    x['answer_range'] = None
    return out


EDITS = [('A', 4, 0, a4_0)]


def main():
    files = {k: os.path.join(FIX, f) for k, f in [('At', 'aclass_qa_test.json'), ('Af', 'aclass_qa_dataset_cleaned.json')]}
    D = {k: json.load(open(f, encoding='utf-8')) for k, f in files.items()}
    log = {}
    for cls, vi, qi, fn in EDITS:
        qid = '%s|test|%d|%d' % (cls, vi, qi)
        tq = D[cls + 't'][vi]['qa_pairs'][qi]
        fq = [q for q in full_qas(cls, D[cls + 'f']) if q['question'] == tq['question']]
        assert len(fq) == 1, (qid, len(fq))
        old = copy.deepcopy(tq)
        outs = [fn(x) for x in (tq, fq[0])]
        assert outs[0] == outs[1]
        log[qid] = {'old_question': old['question'], 'new_question': tq['question'], 'old_answer': old['answer'],
                    'new_answer': tq['answer'], 'answer_range': tq.get('answer_range'), 'question_type': tq['question_type'],
                    'code_output': outs[0], 'changed': sorted(k for k in tq if tq.get(k) != old.get(k))}
        print('==', qid, log[qid]['changed'], '\n   answer:', tq['answer'], '\n   code:', outs[0].replace('\n', ' | '))
    if DRY:
        return
    for k, f in files.items():
        bak = f.replace('.json', '.before_fix8.json')
        if not os.path.exists(bak):
            shutil.copy(f, bak)
        json.dump(D[k], open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(log, open(os.path.join(FIX, 'evaluation', 'fix8_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written; log evaluation/fix8_log.json')


if __name__ == '__main__':
    main()
