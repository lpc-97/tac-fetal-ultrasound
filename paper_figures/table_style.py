"""Presentation pass over the generated table bodies (tables/*.tex), in the style common in vision and medical-imaging journals.
It changes how numbers are typeset and never their value:
  mean\\,$\\pm$\\,std      -> mean\\sd{std}        (standard deviation small and gray, macro in main.tex)
  $+0.45^{*}$ / $-1.55^{*}$ -> \\up{+0.45^{*}} / \\dn{-1.55^{*}}   (gains and losses in colour)
  rows of the proposed method ("(ours)" in the first cell) get \\rowcolor{oursbg}
Idempotent. Called at the end of gen_tables.py and gen_tables_reviewer.py; can also be run on its own."""
import glob
import io
import os
import re

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
PM = re.compile(r"\\,\$\\pm\$\\,(\d+\.\d+)")
DELTA = re.compile(r"\$([+-])(\d+\.\d+)((?:\^\{\*\})?)\$")
OURS = re.compile(r"\(ours\)")  # one rule for every table: rows whose label carries "(ours)"


def restyle(text, shade=OURS):
    out = []
    for line in text.split("\n"):
        line = PM.sub(lambda m: "\\sd{" + m.group(1) + "}", line)
        line = DELTA.sub(lambda m: m.group(0) if float(m.group(2)) == 0 else ("\\up{" if m.group(1) == "+" else "\\dn{") + m.group(1) + m.group(2) + m.group(3) + "}", line)
        first = line.split("&")[0]
        if "&" in line and shade.search(first) and not line.lstrip().startswith("\\rowcolor") and "\\multicolumn" not in first:
            line = "\\rowcolor{oursbg}" + line
        out.append(line)
    return "\n".join(out)


def restyle_dir():
    n = 0
    for p in sorted(glob.glob(os.path.join(TAB, "*.tex"))):
        s = io.open(p, encoding="utf-8", newline="").read()
        t = restyle(s)
        if t != s:
            io.open(p, "w", encoding="utf-8", newline="").write(t); n += 1
    print("table_style: restyled", n, "files")


if __name__ == "__main__":
    restyle_dir()
