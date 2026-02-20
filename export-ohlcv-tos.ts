# Export OHLCV Data from thinkorSwim
#
# HOW TO USE:
# 1. Open /ES chart, 5-minute aggregation, 180 days
# 2. Studies → Edit Studies → Create... → paste this code → OK
# 3. Wait for chart to load (may take a minute)
# 4. At the bottom of thinkorSwim, click the "Messages" tab
# 5. You'll see rows of CSV data. Select ALL (Cmd+A), Copy (Cmd+C)
# 6. Open Terminal, run:  pbpaste > ~/projects/ai-playground/data/es_5m.csv
# 7. Remove this study from your chart when done
#
# Output format: Date/Time,Open,High,Low,Close,Volume

def header = if BarNumber() == 1 then 1 else 0;
AddLabel(header, "Date/Time,Open,High,Low,Close,Volume", Color.WHITE);

def dateVal = GetYYYYMMDD();
def yr = Round(dateVal / 10000, 0);
def mo = Round((dateVal % 10000) / 100, 0);
def dy = dateVal % 100;

def timeVal = GetTime();
def secSinceMidnight = SecondsFromTime(0000);
def hh = Round(secSinceMidnight / 3600, 0);
def mm = Round((secSinceMidnight % 3600) / 60, 0);

# Format: YYYY-MM-DD HH:MM:SS
AddChartBubble(
    no, close,
    yr + "-" +
    (if mo < 10 then "0" else "") + mo + "-" +
    (if dy < 10 then "0" else "") + dy + " " +
    (if hh < 10 then "0" else "") + hh + ":" +
    (if mm < 10 then "0" else "") + mm + ":00" +
    "," + open + "," + high + "," + low + "," + close + "," + volume,
    Color.BLACK
);
