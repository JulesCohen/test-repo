import pandas as pd
import numpy as np

# -------------------
# Utils
# -------------------
def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

def zscore(series, win):
    return (series - series.rolling(win).mean()) / series.rolling(win).std()

def slope(series, win):
    # LinReg slope ~ diff(lr) simplifiée
    lr = series.rolling(win).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    return lr

# -------------------
# Chargement données
# -------------------
# Tu dois fournir un DataFrame `df` daily avec colonnes:
# ['SPY','RSP','IWM','LQD','TLT','HYG','JNK','IEF','VIX','VIX3M']
# index = datetime

# Exemple: df = pd.read_csv("data.csv", parse_dates=['Date'], index_col='Date')

# -------------------
# Paramètres
# -------------------
scale = 15   # ~3W = 15 daily bars
lenMA = 10 * scale
slopeLen = 3 * scale
pWin = 34 * scale
pers = 2 * scale
creditThresh = -0.45

# -------------------
# Ratios
# -------------------
df['RS_RSP'] = safe_div(df['RSP'], df['SPY'])
df['RS_IWM'] = safe_div(df['IWM'], df['SPY'])
df['RD']     = safe_div(df['LQD'], df['TLT'])
df['RS_HL']  = safe_div(df['HYG'], df['LQD'])
df['RS_JI']  = safe_div(df['JNK'], df['IEF'])
df['TERM']   = safe_div(df['VIX3M'], df['VIX'])

# -------------------
# EARLY votes
# -------------------
df['EV_RSP']  = (df['RS_RSP'] < df['RS_RSP'].rolling(lenMA).mean()) & (slope(df['RS_RSP'], slopeLen) < 0)
df['EV_IWM']  = (df['RS_IWM'] < df['RS_IWM'].rolling(lenMA).mean()) & (slope(df['RS_IWM'], slopeLen) < 0)
df['EV_TERM'] = (df['TERM']   < 1.0)

df['Votes']   = df[['EV_RSP','EV_IWM','EV_TERM']].sum(axis=1)
df['Avail']   = df[['RS_RSP','RS_IWM','TERM']].notna().sum(axis=1)
df['EARLY']   = (df['Avail']>=2) & (df['Votes']>=2)

# -------------------
# Confirmations
# -------------------
# Duration
df['Cond_DUR'] = (df['RD'] < df['RD'].rolling(lenMA).mean()) & (slope(df['RD'], slopeLen) < 0)
df['OK_DUR']   = df['Cond_DUR'].rolling(pers).sum() >= pers

# Crédit
df['Z_HL'] = zscore(df['RS_HL'], pWin)
df['Z_JI'] = zscore(df['RS_JI'], pWin)
df['CREDIT_BAD'] = (
    ((df['Z_HL'] < creditThresh) | (df['Z_JI'] < creditThresh))
    & ((slope(df['RS_HL'], slopeLen) < 0) | (slope(df['RS_JI'], slopeLen) < 0))
)
df['OK_CRED'] = df['CREDIT_BAD'].rolling(pers).sum() >= pers

# Breadth fallback
df['Z_RS'] = zscore(df['RS_RSP'], pWin)
df['BREADTH_BAD'] = (df['Z_RS'] < (creditThresh+0.10)) & (slope(df['RS_RSP'], slopeLen) < 0)
df['OK_BREADTH'] = df['BREADTH_BAD'].rolling(pers).sum() >= pers

df['CONFIRM'] = df['OK_DUR'] | df['OK_CRED'] | df['OK_BREADTH']

# -------------------
# Fenêtre de confirmation
# -------------------
df['Z_VIX'] = zscore(df['VIX'], pWin)
df['ConfWin'] = np.where(df['Z_VIX']>0.5, 5*scale, np.where(df['Z_VIX']>-0.2, 4*scale, 3*scale))

# -------------------
# Machine à états
# -------------------
df['Signal'] = False
armed = False
armBar = None

for i in range(len(df)):
    if not armed and df['EARLY'].iloc[i]:
        armed = True
        armBar = i
    elif armed:
        within = (i - armBar) <= df['ConfWin'].iloc[i]
        lead_ok = (i - armBar) >= scale
        if within and lead_ok and df['CONFIRM'].iloc[i]:
            df.at[df.index[i],'Signal'] = True
            armed = False
        elif not within:
            armed = False

# -------------------
# Résultats
# -------------------
signals = df[df['Signal']]
print(signals[['EARLY','CONFIRM']])
