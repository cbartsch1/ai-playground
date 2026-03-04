declare lower;

input avgLength = 20;

def avgVol = Average(volume, avgLength);
def isAbove = volume > avgVol;
def pct = Round(((volume - avgVol) / avgVol) * 100, 1);

plot Vol = volume;
Vol.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Vol.AssignValueColor(if isAbove then Color.WHITE else Color.GRAY);

plot Avg = avgVol;
Avg.SetDefaultColor(Color.YELLOW);
Avg.SetLineWeight(2);

plot PctVsAvg = pct;
PctVsAvg.SetDefaultColor(Color.CYAN);
PctVsAvg.SetLineWeight(1);
PctVsAvg.HideBubble();

AddLabel(yes, "vs Avg: " + (if pct >= 0 then "+" else "") + pct + "%",
    if isAbove then Color.WHITE else Color.GRAY);
