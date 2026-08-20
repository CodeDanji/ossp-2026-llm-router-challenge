# PromptBudget Dev policy comparison

All values are aggregate public-Dev measurements; no prompt, ID, or outcome row is included.

| Candidate | Fast cost / margin / score / distribution | Balanced cost / margin / score / distribution | Premium cost / margin / score / distribution | Weighted score | Runtime | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| all-light | 1 / 0.25 / 0.619318181818 / {"ax31":0,"ax31-light":880,"axk1-think":0} | 1 / 1.0 / 0.619318181818 / {"ax31":0,"ax31-light":880,"axk1-think":0} | 1 / 3.0 / 0.619318181818 / {"ax31":0,"ax31-light":880,"axk1-think":0} | 0.619318181818 | stdlib | pass |
| official-prompt-heuristic | 1.072334054331 / 0.177665945669 / 0.625852272727 / {"ax31":127,"ax31-light":753,"axk1-think":0} | 1.367865816807 / 0.632134183193 / 0.658238636364 / {"ax31":393,"ax31-light":487,"axk1-think":0} | 2.102044148164 / 1.897955851836 / 0.691761363636 / {"ax31":880,"ax31-light":0,"axk1-think":0} | 0.655340909091 | stdlib | pass |
| absolute-linear | 1.057062675456 / 0.192937324544 / 0.619886363636 / {"ax31":2,"ax31-light":877,"axk1-think":1} | 1.760065999715 / 0.239934000285 / 0.691477272727 / {"ax31":619,"ax31-light":257,"axk1-think":4} | 3.62388313171 / 0.37611686829 / 0.715625 / {"ax31":493,"ax31-light":201,"axk1-think":186} | 0.670085227273 | stdlib | pass |
| delta-linear | — | — | — | — | — | deferred |
| sparse-knn | — | — | — | — | — | deferred |
