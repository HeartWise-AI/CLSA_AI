"""Traditional CVD risk equations (10-year risk), vectorised, with unit tests against published examples.

FRS 2008  : D'Agostino RB et al. Circulation 2008;117:743-753 (general CVD, lipid version, Cox model).
PCE 2013  : Goff DC et al. 2013 ACC/AHA risk assessment guideline (Pooled Cohort Equations, hard ASCVD);
            Black participants -> African-American equations, all other groups -> White equations.
Lipids in mg/dL (mmol/L x 38.67). Ages outside the validated range are flagged by the callers, not clipped here.
"""
import numpy as np

MMOL_TO_MGDL = 38.67


def frs2008(age, female, tc, hdl, sbp, bp_treated, smoker, diabetes):
    age, tc, hdl, sbp = map(np.asarray, (age, tc, hdl, sbp))
    female, bp_treated, smoker, diabetes = (np.asarray(x).astype(bool) for x in (female, bp_treated, smoker, diabetes))
    la, ltc, lhdl, lsbp = np.log(age), np.log(tc), np.log(hdl), np.log(sbp)
    w = (2.32888 * la + 1.20904 * ltc - 0.70833 * lhdl + np.where(bp_treated, 2.82263, 2.76157) * lsbp
         + 0.52873 * smoker + 0.69154 * diabetes)
    m = (3.06117 * la + 1.12370 * ltc - 0.93263 * lhdl + np.where(bp_treated, 1.99881, 1.93303) * lsbp
         + 0.65451 * smoker + 0.57367 * diabetes)
    return np.where(female, 1 - 0.95012 ** np.exp(w - 26.1931), 1 - 0.88936 ** np.exp(m - 23.9802))


def frs2008_bmi(age, female, bmi, sbp, bp_treated, smoker, diabetes):
    """D'Agostino 2008 office-based (non-laboratory) general-CVD model: BMI replaces TC and HDL."""
    age, bmi, sbp = map(np.asarray, (age, bmi, sbp))
    female, bp_treated, smoker, diabetes = (np.asarray(x).astype(bool) for x in (female, bp_treated, smoker, diabetes))
    la, lb, lsbp = np.log(age), np.log(bmi), np.log(sbp)
    w = 2.72107 * la + 0.51125 * lb + np.where(bp_treated, 2.88267, 2.81291) * lsbp + 0.61868 * smoker + 0.77763 * diabetes
    m = 3.11296 * la + 0.79277 * lb + np.where(bp_treated, 1.92672, 1.85508) * lsbp + 0.70953 * smoker + 0.53160 * diabetes
    return np.where(female, 1 - 0.94833 ** np.exp(w - 26.0145), 1 - 0.88431 ** np.exp(m - 23.9388))


def pce2013(age, female, black, tc, hdl, sbp, bp_treated, smoker, diabetes):
    age, tc, hdl, sbp = map(np.asarray, (age, tc, hdl, sbp))
    female, black, bpt, smk, dm = (np.asarray(x).astype(bool) for x in (female, black, bp_treated, smoker, diabetes))
    la, ltc, lhdl, lsbp = np.log(age), np.log(tc), np.log(hdl), np.log(sbp)
    # White female
    wf = (-29.799 * la + 4.884 * la ** 2 + 13.540 * ltc - 3.114 * la * ltc - 13.578 * lhdl + 3.149 * la * lhdl
          + np.where(bpt, 2.019, 1.957) * lsbp + 7.574 * smk - 1.665 * la * smk + 0.661 * dm)
    # African-American female
    af = (17.114 * la + 0.940 * ltc - 18.920 * lhdl + 4.475 * la * lhdl
          + np.where(bpt, 29.291 * lsbp - 6.432 * la * lsbp, 27.820 * lsbp - 6.087 * la * lsbp)
          + 0.691 * smk + 0.874 * dm)
    # White male
    wm = (12.344 * la + 11.853 * ltc - 2.664 * la * ltc - 7.990 * lhdl + 1.769 * la * lhdl
          + np.where(bpt, 1.797, 1.764) * lsbp + 7.837 * smk - 1.795 * la * smk + 0.658 * dm)
    # African-American male
    am = 2.469 * la + 0.302 * ltc - 0.307 * lhdl + np.where(bpt, 1.916, 1.809) * lsbp + 0.549 * smk + 0.645 * dm
    r_wf = 1 - 0.9665 ** np.exp(wf + 29.18)
    r_af = 1 - 0.9533 ** np.exp(af - 86.61)
    r_wm = 1 - 0.9144 ** np.exp(wm - 61.18)
    r_am = 1 - 0.8954 ** np.exp(am - 19.54)
    return np.where(female, np.where(black, r_af, r_wf), np.where(black, r_am, r_wm))


def ten_to_five(r10):
    """S(5) = S(10)**0.5: assumes a CONSTANT BASELINE HAZARD over 10 y (exponential survival)."""
    return 1 - (1 - np.asarray(r10)) ** 0.5


def _tests():
    # Goff 2013, Table A example: age 55, TC 213, HDL 50, SBP 120 untreated, non-smoker, no diabetes
    ex = dict(age=55, tc=213, hdl=50, sbp=120, bp_treated=0, smoker=0, diabetes=0)
    got = {k: float(pce2013(female=f, black=b, **ex)) for k, f, b in
           [("WF", 1, 0), ("AAF", 1, 1), ("WM", 0, 0), ("AAM", 0, 1)]}
    exp = {"WF": 0.021, "AAF": 0.030, "WM": 0.053, "AAM": 0.061}
    for k in exp:
        assert abs(got[k] - exp[k]) < 0.0015, (k, got[k], exp[k])
    # D'Agostino 2008 worked examples
    f = float(frs2008(age=61, female=1, tc=180, hdl=47, sbp=124, bp_treated=0, smoker=1, diabetes=0))
    m = float(frs2008(age=53, female=0, tc=161, hdl=55, sbp=125, bp_treated=1, smoker=0, diabetes=1))
    assert abs(f - 0.1048) < 0.002, f
    assert abs(m - 0.1565) < 0.002, m
    print("risk_scores unit tests passed:", {k: round(v, 4) for k, v in got.items()}, "FRS", round(f, 4), round(m, 4))


if __name__ == "__main__":
    _tests()
