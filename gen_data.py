"""
Churn Signal Finder — synthetic user dataset generator.

The dataset is generated from an EXPLICIT churn model so the analysis has a
known ground truth to recover. This is deliberate: it lets us verify the
segmentation method actually finds the planted effect and, just as important,
does NOT flag the cells where no effect exists.

Churn is drawn from a logistic model:
    logit(churn) = b0
                 + b_engage * (active_days_90 standardized)
                 + b_tickets * support_tickets
                 + b_apac_business * 1[plan=Business & region=APAC & signup>2024-09]

The final term is the planted "product defect" signal we want the analysis to find.
All coefficients are recorded in ground_truth.json for later verification.
"""
import numpy as np, pandas as pd, json
from datetime import datetime, timedelta

np.random.seed(42)
N, start, span_days = 12000, datetime(2024,1,1), 540

# --- model coefficients (the ground truth) ---
COEF = dict(
    intercept=-2.1,          # baseline log-odds of churn
    b_engage=-0.55,          # more engagement -> less churn
    b_tickets=0.28,          # more support tickets -> more churn
    b_apac_business=1.15,    # THE PLANTED DEFECT: extra churn log-odds for the pocket
    defect_start="2024-09-01"
)

plans=['Free','Starter','Business','Enterprise']; plan_w=[0.45,0.30,0.18,0.07]
regions=['NA','EMEA','APAC','LATAM']; region_w=[0.40,0.30,0.20,0.10]
channels=['Organic','Paid','Referral','Partner']; chan_w=[0.5,0.25,0.15,0.10]
segments=['SMB','Mid-Market','Enterprise']; seg_w=[0.6,0.28,0.12]
price={'Free':0,'Starter':12,'Business':29,'Enterprise':49}
base_active={'Free':6,'Starter':14,'Business':20,'Enterprise':26}
base_tix={'Free':0.3,'Starter':0.8,'Business':1.5,'Enterprise':2.4}
defect_start=datetime.strptime(COEF['defect_start'],"%Y-%m-%d")

def sigmoid(x): return 1/(1+np.exp(-x))

rows=[]
for i in range(N):
    signup=start+timedelta(days=int(np.random.uniform(0,span_days)))
    plan=np.random.choice(plans,p=plan_w)
    region=np.random.choice(regions,p=region_w)
    channel=np.random.choice(channels,p=chan_w)
    seg=np.random.choice(segments,p=seg_w)
    seats={'Free':1,'Starter':int(np.random.uniform(2,8)),
           'Business':int(np.random.uniform(5,40)),
           'Enterprise':int(np.random.uniform(30,300))}[plan]
    mrr=price[plan]*seats
    active=max(0,int(np.random.normal(base_active[plan],7)))
    in_pocket=(plan=='Business' and region=='APAC' and signup>defect_start)
    tickets=np.random.poisson(base_tix[plan]+(2.0 if in_pocket else 0))

    # churn from the logistic model
    z=(active-15)/8.0
    logit=(COEF['intercept']+COEF['b_engage']*z+COEF['b_tickets']*tickets
           +(COEF['b_apac_business'] if in_pocket else 0))
    churned=int(np.random.random()<sigmoid(logit))

    csat=None
    if tickets>0:
        base=4.3-tickets*0.15-(1.4 if in_pocket else 0)
        csat=round(min(5,max(1,np.random.normal(base,0.7))),1)

    rows.append(dict(user_id=f"U{100000+i}",signup_date=signup.date(),plan=plan,
        region=region,acquisition_channel=channel,segment=seg,seats=seats,mrr=mrr,
        active_days_90=active,support_tickets=tickets,csat=csat,churned=churned))

df=pd.DataFrame(rows)
# realistic mess for the cleaning step to handle
df.loc[df.sample(frac=0.03,random_state=1).index,'region']=None
df.loc[df.sample(frac=0.02,random_state=2).index,'active_days_90']=None
df.loc[df.sample(frac=0.015,random_state=3).index,'plan']=df['plan'].str.lower()
df=pd.concat([df,df.sample(80,random_state=4)],ignore_index=True)
df.to_csv('churn_users_raw.csv',index=False)
json.dump(COEF,open('ground_truth.json','w'),indent=2)
print(f"Generated {len(df)} rows. Planted APAC-Business defect coef = {COEF['b_apac_business']}")
