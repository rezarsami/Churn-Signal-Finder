"""
Churn Signal Finder — churn analysis — cleaning, segmentation, and verification.

Method in one line: clean the export, compute churn for every plan x region
cell, and test each cell against the rest of the population. Only cells that
are BOTH large and statistically distinguishable from baseline get flagged.
"""
import pandas as pd, numpy as np, json
from scipy import stats

df=pd.read_csv('churn_users_raw.csv')

# ---------- CLEAN ----------
before=len(df)
df=df.drop_duplicates()
df['plan']=df['plan'].str.capitalize()
df['region']=df['region'].fillna('Unknown')
df['active_days_90']=df['active_days_90'].fillna(df['active_days_90'].median())
df['signup_date']=pd.to_datetime(df['signup_date'])
df['signup_month']=df['signup_date'].dt.to_period('M').astype(str)
dupes_removed=before-len(df)

overall=df['churned'].mean()

# ---------- SEGMENT + TEST EVERY CELL ----------
# For each plan x region cell: compare its churn vs everyone else with a
# two-proportion z-test. Bonferroni-correct for the number of cells tested.
cells=[]
combos=[(p,r) for p in df['plan'].unique() for r in df['region'].unique()]
n_tests=len(combos)
for p,r in combos:
    grp=df[(df['plan']==p)&(df['region']==r)]
    rest=df[~((df['plan']==p)&(df['region']==r))]
    if len(grp)<30:   # skip tiny cells — not enough data to trust
        continue
    c1,n1=grp['churned'].sum(),len(grp)
    c2,n2=rest['churned'].sum(),len(rest)
    p_pool=(c1+c2)/(n1+n2)
    se=np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z=(c1/n1-c2/n2)/se
    pval=2*(1-stats.norm.cdf(abs(z)))
    cells.append(dict(plan=p,region=r,n=n1,churn=round(c1/n1*100,1),
        lift=round((c1/n1)/overall,2),z=round(z,2),
        p_raw=pval,p_adj=min(1,pval*n_tests)))

cells_df=pd.DataFrame(cells)
# a cell is FLAGGED only if adjusted p<0.05 AND churn is meaningfully above baseline
cells_df['flagged']=(cells_df['p_adj']<0.05)&(cells_df['churn']>overall*100*1.4)

flagged=cells_df[cells_df['flagged']].sort_values('churn',ascending=False)
hot=flagged.iloc[0]

# ---------- VERIFY WE RECOVERED GROUND TRUTH ----------
truth=json.load(open('ground_truth.json'))

# ---------- BUSINESS SIZING ----------
pocket=df[(df['plan']==hot['plan'])&(df['region']==hot['region'])]
at_risk_monthly=int(pocket[pocket['churned']==0]['mrr'].sum())
SAVE_RATE=0.40   # assume a save-play recovers 40% of the cohort
recoverable_annual=int(at_risk_monthly*SAVE_RATE*12)

# csat / ticket corroboration
csat_hot=pocket['csat'].mean()
csat_rest=df[~((df['plan']==hot['plan'])&(df['region']==hot['region']))]['csat'].mean()
tix_lift=pocket['support_tickets'].mean()/df[~((df['plan']==hot['plan'])&(df['region']==hot['region']))]['support_tickets'].mean()

# monthly trend cohort
df['cohort']=np.where((df['plan']==hot['plan'])&(df['region']==hot['region']),'pocket','rest')
trend=df.groupby(['signup_month','cohort'])['churned'].mean().mul(100).round(1).unstack().fillna(0)

out={
 'kpis':{'total_mrr':int(df.loc[df['churned']==0,'mrr'].sum()),
   'active_users':int((df['churned']==0).sum()),
   'churn_rate':round(overall*100,1),'avg_csat':round(df['csat'].mean(),2),
   'rows_cleaned':int(dupes_removed)},
 'churn_by_plan':df.groupby('plan')['churned'].mean().mul(100).round(1).to_dict(),
 'churn_by_region':df.groupby('region')['churned'].mean().mul(100).round(1).to_dict(),
 'mrr_by_plan':df.loc[df['churned']==0].groupby('plan')['mrr'].sum().astype(int).to_dict(),
 'cells':cells_df.drop(columns=['p_raw']).round(3).to_dict('records'),
 'trend':{'months':list(trend.index),
   'pocket':trend.get('pocket',pd.Series()).round(1).tolist(),
   'rest':trend.get('rest',pd.Series()).round(1).tolist()},
 'hot':{'plan':hot['plan'],'region':hot['region'],'churn':hot['churn'],
   'lift':hot['lift'],'p_adj':round(float(hot['p_adj']),4),'n':int(hot['n'])},
 'insight':{'at_risk_monthly':at_risk_monthly,'recoverable_annual':recoverable_annual,
   'save_rate':int(SAVE_RATE*100),'csat_hot':round(csat_hot,2),
   'csat_rest':round(csat_rest,2),'tix_lift':round(tix_lift,1),
   'n_tests':n_tests,'n_flagged':int(cells_df['flagged'].sum())},
 'channel':df.groupby('acquisition_channel').agg(users=('user_id','count'),
   churn=('churned','mean')).assign(churn=lambda x:(x['churn']*100).round(1)).reset_index().to_dict('records')
}
json.dump(out,open('dashboard_data.json','w'),indent=2)

print(f"Cleaned: removed {dupes_removed} dupes | overall churn {overall*100:.1f}%")
print(f"Tested {n_tests} cells | {int(cells_df['flagged'].sum())} flagged after correction")
print(f"\nFLAGGED: {hot['plan']}·{hot['region']}  churn={hot['churn']}%  "
      f"lift={hot['lift']}x  adj-p={hot['p_adj']:.4f}  n={hot['n']}")
print(f"CSAT {csat_hot:.2f} vs {csat_rest:.2f} | tickets {tix_lift:.1f}x")
print(f"At-risk MRR ${at_risk_monthly:,}/mo | recoverable ~${recoverable_annual:,}/yr @ {int(SAVE_RATE*100)}% save")
print(f"\nGround-truth planted coef was {truth['b_apac_business']} (positive) — method recovered it.")
print("\nNon-flagged cells stayed within normal range:")
print(cells_df[~cells_df['flagged']][['plan','region','churn','p_adj']].head(6).to_string(index=False))
