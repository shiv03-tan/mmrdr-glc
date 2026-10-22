# Usage:  python count_violations.py path\to\train_manifest.csv
# Counts the 22 annotation-rule violations and the number of UNIQUE images behind them.
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
d = d[d["image"] != "img/tr000215.jpg"]          # uncertain-IH record excluded, as in training
L7 = ["MA","HE","IH","VB_IRMA","NV","VH","RD"]; ADV = ["NV","VH","RD"]
anyL = d[L7].sum(axis=1) > 0; adv = d[ADV].sum(axis=1) > 0
c = {"grade0_any_lesion": (d.grade == 0) & anyL,
     "grade_positive_no_lesion": (d.grade > 0) & ~anyL,
     "below4_advanced": (d.grade < 4) & adv,
     "grade4_no_advanced": (d.grade == 4) & ~adv}
for k, m in c.items(): print(k, int(m.sum()))
u = (c["grade0_any_lesion"] | c["grade_positive_no_lesion"] | c["below4_advanced"] | c["grade4_no_advanced"])
print("total category counts:", sum(int(m.sum()) for m in c.values()))
print("unique images with >=1 violation:", int(u.sum()), "of", len(d))
