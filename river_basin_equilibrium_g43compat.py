#Import GAMS API and other dependencies
import gams
import os
import sys
import itertools
import numpy as np
import pandas as pd

#Define model in GAMS syntax
def get_model_text():
    return """option limrow=10000;

Sets
i players in the river basin
c consumptive use classes
t time periods
pt(t) time periods within planning horizon
v variables with starting points to vary
;

*Input parameters provided by user
Parameters
cyc subset of time periods in planning horizon
prd period of time between time periods
ir interest rate
beta inverse demand curve slope
n(i) river basin inflow for player i
rfc(i) regulatory flow constraint limiting player i withdrawals
c_ops(i) operational unit cost for player i
c_sr(i) supply-related unit cost for player i
c_cap(i) capital costs for each player
c_cu(i,c) consumptive use reduction costs for player i in class c
a_req(i,t) capital improvement requirement for node i at time t
demand(i,t) water demand for player i in time period t
lf(c,i,t) fraction of losses for player i in time period t for class c
srt_pts(v) solution starting points
;

*import data from gdx file
$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load i c t v prd cyc ir beta n rfc c_ops c_sr c_cap c_cu a_req demand lf srt_pts
$gdxin

Alias(i,j,k_in);
Alias(t,tt,ttt,t4);

*Specify which time periods are in the planning horizon
pt(t)$(ord(t)le cyc) = yes;

*Parameters calculated from input parameters
Parameters
alpha(i,t) Inverse demand intercept
d(t) Discount rate for each time period
delta_US_all(i,i) References all upstream neighbors for each utility
delta_DS_all(i,i) References all downstream neighbors for each utility
delta_F(t,t) Forward time periods
delta_Fml(t,t) Forward time periods minus last
delta_CAP(t,t) Time periods where capital project takes effect
delta_B(t,t) Backwards time periods including present
delta_BP(t,t) Backward time periods only including past
;

*Assign values to parameters
d(t) = (1+ir)**(-prd*(ord(t)-1));

delta_US_all(j,i)$ (ord(j) lt ord(i)) = 1;

delta_DS_all(k_in,i)$ (ord(k_in) gt ord(i)) = 1;

delta_F(tt,t) $ (ord(tt) ge ord(t)) = 1;

delta_Fml(tt,t) $ (ord(tt) ge ord(t)) = 1;
delta_Fml(tt,t) $ (ord(tt) = card(t)) = 0;

delta_CAP(tt,t) $ ((ord(tt) ge ord(t)) or (ord(tt) ge card(pt))) = 1;

delta_B(tt,t) $ (ord(tt) le ord(t)) = 1;

delta_BP(tt,t) $ (ord(tt) lt ord(t)) = 1;

alpha(i,t) = c_ops(i) + beta*demand(i,t);

Positive Variables
WD(i,t) Incremental direct water withdrawals for player i in time period t
WS(i,t) Water supply releases for player i in time period t
Q(i,t) Total player i water consumption in time period t
K(i,t) Capital improvements for player i in time period t
LR(i,c,t) Class c loss reductions for player i in time period t
WP_S(i,t) Player i's water purchases in the cost share market for time period t
WP_G(i,j,t) Player i's water purchases in the general commodity market for time period t
GMA_loss(i,c,t) Gamma loss shadow prices
GMA_flow(i,t) Gamma flow shadow prices
GMA_cap(i,t) Gamma cap shadow prices
PI_as(i,t) Water purchase price for player i in time period t
;

*Players cannot buy water from themselves or downstream players
WP_G.fx(i,j,t) $(delta_US_all(j,i) ne 1) = 0;
WP_G.fx(i,i,t) = 0;
Variables
O_min(i,t) Minimum outflow for player i in time period t
LMB_sup(i,t) Lambda supply shadow prices
LMB_aug(i,t) Lambda augment shadow prices
THT(i,t) Inverse demand curve price
;

*Define variable starting points
WD.l(i,t) = srt_pts('1');
WS.l(i,t) = srt_pts('2');
Q.l(i,t) = srt_pts('3');
K.l(i,t) = srt_pts('4');
LR.l(i,c,t) = srt_pts('5');
WP_S.l(i,t) = srt_pts('6');
WP_G.l(i,j,t) = srt_pts('7');
GMA_loss.l(i,c,t) = srt_pts('8');
GMA_flow.l(i,t) = srt_pts('9');
GMA_cap.l(i,t) = srt_pts('10');
PI_as.l(i,t) = srt_pts('11');
O_min.l(i,t) = srt_pts('12'); 
LMB_sup.l(i,t) = srt_pts('13');
LMB_aug.l(i,t) = srt_pts('14');
THT.l(i,t) = srt_pts('15');

Equations
GCM_KKT_STAT_WD(i,t) KKT Stationary conditions for general commodity market
GCM_KKT_STAT_WS(i,t) KKT Stationary conditions for general commodity market
GCM_KKT_STAT_Q(i,t) KKT Stationary conditions for general commodity market
GCM_KKT_STAT_K(i,t) KKT Stationary conditions for general commodity market
GCM_KKT_STAT_LR(i,c,t) KKT Stationary conditions for general commodity market
GCM_KKT_STAT_WP(i,j,t) KKT Stationary conditions for general commodity market
GCM_KKT_FEAS_LOSS(i,c,t) KKT Feasibility conditions for general commodity market
GCM_KKT_FEAS_FLOW(i,t) KKT Feasibility conditions for general commodity market
GCM_KKT_FEAS_CAP(i,t) KKT Feasibility conditions for general commodity market
GCM_SYSTEM_MC(i,t) Market clearing conditions for general commodity market
GCM_KKT_FEAS_SUP(i,t) KKT Feasibility conditions for general commodity market
GCM_KKT_FEAS_AUG(i,t) KKT Feasibility conditions for general commodity market
GCM_SYSTEM_OUTFLOW(i,t) MCP system constraint for minimum outflow
GCM_SYSTEM_THETA(i,t) Inverse demand curves

CSM_KKT_STAT_WD(i,t) KKT Stationary conditions for cost sharing market
CSM_KKT_STAT_WS(i,t) KKT Stationary conditions for cost sharing market
CSM_KKT_STAT_Q(i,t) KKT Stationary conditions for cost sharing market
CSM_KKT_STAT_K(i,t) KKT Stationary conditions for cost sharing market
CSM_KKT_STAT_LR(i,c,t) KKT Stationary conditions for cost sharing market
CSM_KKT_STAT_WP(i,t) KKT Stationary conditions for cost sharing market
CSM_KKT_FEAS_LOSS(i,c,t) KKT Feasibility conditions for cost sharing market
CSM_KKT_FEAS_FLOW(i,t) KKT Feasibility conditions for cost sharing market
CSM_KKT_FEAS_CAP(i,t) KKT Feasibility conditions for cost sharing market
CSM_SYSTEM_MC(i,t) Market clearing conditions for cost sharing market
CSM_KKT_FEAS_SUP(i,t) KKT Feasibility conditions for cost sharing market
CSM_KKT_FEAS_AUG(i,t) KKT Feasibility conditions for cost sharing market
CSM_SYSTEM_OUTFLOW(i,t) MCP system constraint for minimum outflow
CSM_SYSTEM_THETA(i,t) Inverse demand curves

Equations
NM_KKT_STAT_WD(i,t) KKT Stationary conditions for no market
NM_KKT_STAT_WS(i,t) KKT Stationary conditions for no market
NM_KKT_STAT_Q(i,t) KKT Stationary conditions for no market
NM_KKT_STAT_K(i,t) KKT Stationary conditions for no market
NM_KKT_FEAS_FLOW(i,t) KKT Feasibility conditions for no market
NM_KKT_FEAS_CAP(i,t) KKT Feasibility conditions for no market
NM_KKT_FEAS_SUP(i,t) KKT Feasibility conditions for no market
NM_KKT_FEAS_AUG(i,t) KKT Feasibility conditions for no market
NM_SYSTEM_OUTFLOW(i,t) MCP system constraint for minimum outflow
NM_SYSTEM_THETA(i,t) Inverse demand curves
;

GCM_KKT_STAT_WD(i,t).. sum(tt,delta_F(tt,t)*(LMB_sup(i,tt)-sum(c, lf(c,i,t)*GMA_loss(i,c,tt))))=g=0;
GCM_KKT_STAT_WS(i,t).. d(t)*(c_sr(i)-PI_as(i,t))-GMA_flow(i,t)+GMA_cap(i,t)=g=0;
GCM_KKT_STAT_Q(i,t).. d(t)*(c_ops(i)-THT(i,t))-LMB_sup(i,t)+GMA_flow(i,t)=g=0;
GCM_KKT_STAT_K(i,t).. d(t)*c_cap(i)-GMA_cap(i,t)+LMB_aug(i,t)=g=0;
GCM_KKT_STAT_LR(i,c,t).. d(t)*(c_cu(i,c)-PI_as(i,t))+sum(tt,delta_F(tt,t)*GMA_loss(i,c,tt))=g=0;
GCM_KKT_STAT_WP(i,j,t).. delta_US_all(j,i)*(d(t)*PI_as(j,t)-GMA_flow(i,t))=g=0;
GCM_KKT_FEAS_LOSS(i,c,t).. sum(tt, delta_B(tt,t)*(lf(c,i,tt)*WD(i,tt)-LR(i,c,tt)))=g=0;
GCM_KKT_FEAS_FLOW(i,t).. n(i)+WS(i,t)+sum(j,delta_US_all(j,i)*WP_G(i,j,t))-rfc(i)+O_min(i-1,t)-Q(i,t)=g=0;
GCM_KKT_FEAS_CAP(i,t).. K(i,t)-WS(i,t)=g=0;
GCM_SYSTEM_MC(i,t).. sum(c,LR(i,c,t))+WS(i,t)- sum(k_in, delta_DS_all(k_in,i)*WP_G(k_in,i,t))=g=0;
GCM_KKT_FEAS_SUP(i,t).. sum(tt, delta_B(tt,t)*WD(i,tt))-Q(i,t)=e=0;
GCM_KKT_FEAS_AUG(i,t).. K(i,t) - a_req(i,t)=e=0;
GCM_SYSTEM_OUTFLOW(i,t).. O_min(i,t)=e=n(i)-sum((c,tt), delta_B(tt,t)*lf(c,i,tt)*WD(i,tt)-delta_BP(tt,t)*LR(i,c,tt))+O_min(i-1,t);
GCM_SYSTEM_THETA(i,t).. THT(i,t)=e=alpha(i,t)-beta*Q(i,t);

CSM_KKT_STAT_WD(i,t).. sum(tt,delta_F(tt,t)*(LMB_sup(i,tt)-sum(c, lf(c,i,t)*GMA_loss(i,c,tt))))=g=0;
CSM_KKT_STAT_WS(i,t).. d(t)*(c_sr(i)-sum(k_in,PI_as(k_in,t)*delta_DS_all(k_in,i)))-GMA_flow(i,t)+GMA_cap(i,t)=g=0;
CSM_KKT_STAT_Q(i,t).. d(t)*(c_ops(i)-THT(i,t))-LMB_sup(i,t)+GMA_flow(i,t)=g=0;
CSM_KKT_STAT_K(i,t).. d(t)*c_cap(i)-GMA_cap(i,t)+LMB_aug(i,t)=g=0;
CSM_KKT_STAT_LR(i,c,t).. d(t)*(c_cu(i,c)-sum(k_in, PI_as(k_in,t)*delta_DS_all(k_in,i)))+sum(tt,GMA_loss(i,c,tt)*delta_F(tt,t))=g=0;
CSM_KKT_STAT_WP(i,t).. d(t)*PI_as(i,t)-GMA_flow(i,t)=g=0;
CSM_KKT_FEAS_LOSS(i,c,t)..sum(tt,delta_B(tt,t)*(lf(c,i,tt)*WD(i,tt)-LR(i,c,tt)))=g=0;
CSM_KKT_FEAS_FLOW(i,t).. n(i)+WS(i,t)+WP_S(i,t)-rfc(i)+O_min(i-1,t)-Q(i,t)=g=0;
CSM_KKT_FEAS_CAP(i,t).. K(i,t)-WS(i,t)=g=0;
CSM_SYSTEM_MC(i,t).. sum(j, delta_US_all(j,i)*(sum(c,LR(j,c,t))+WS(j,t)))-WP_S(i,t) =g=0;
CSM_KKT_FEAS_SUP(i,t).. sum(tt, delta_B(tt,t)*WD(i,tt))-Q(i,t)=e=0;
CSM_KKT_FEAS_AUG(i,t).. K(i,t) - a_req(i,t)=e=0;
CSM_SYSTEM_OUTFLOW(i,t).. O_min(i,t)=e=n(i)-sum((c,tt), delta_B(tt,t)*lf(c,i,tt)*WD(i,tt)-delta_BP(tt,t)*LR(i,c,tt))+O_min(i-1,t);
CSM_SYSTEM_THETA(i,t).. THT(i,t)=e=alpha(i,t)-beta*Q(i,t);

NM_KKT_STAT_WD(i,t).. sum(tt,delta_F(tt,t)*(LMB_sup(i,tt)))=g=0;
NM_KKT_STAT_WS(i,t).. d(t)*c_sr(i)-GMA_flow(i,t)+GMA_cap(i,t)=g=0;
NM_KKT_STAT_Q(i,t).. d(t)*(c_ops(i)-THT(i,t))-LMB_sup(i,t)+GMA_flow(i,t)=g=0;
NM_KKT_STAT_K(i,t).. d(t)*c_cap(i)-GMA_cap(i,t)+LMB_aug(i,t)=g=0;
NM_KKT_FEAS_FLOW(i,t).. n(i)+WS(i,t)-rfc(i)+O_min(i-1,t)-Q(i,t)=g=0;
NM_KKT_FEAS_CAP(i,t).. K(i,t)-WS(i,t)=g=0;
NM_KKT_FEAS_SUP(i,t).. sum(tt, delta_B(tt,t)*WD(i,tt))-Q(i,t)=e=0;
NM_KKT_FEAS_AUG(i,t).. K(i,t) - a_req(i,t)=e=0;
NM_SYSTEM_OUTFLOW(i,t).. O_min(i,t)=e=n(i)-sum((c,tt), delta_B(tt,t)*lf(c,i,tt)*WD(i,tt))+O_min(i-1,t);
NM_SYSTEM_THETA(i,t).. THT(i,t)=e=alpha(i,t)-beta*Q(i,t);

Model DRA_Access_Asymmetry_Gen_Cmdty_Mkt
/
GCM_KKT_STAT_WD.WD
GCM_KKT_STAT_WS.WS
GCM_KKT_STAT_Q.Q
GCM_KKT_STAT_K.K
GCM_KKT_STAT_LR.LR
GCM_KKT_STAT_WP.WP_G
GCM_KKT_FEAS_LOSS.GMA_loss
GCM_KKT_FEAS_FLOW.GMA_flow
GCM_KKT_FEAS_CAP.GMA_cap
GCM_SYSTEM_MC.PI_as
GCM_KKT_FEAS_SUP
GCM_KKT_FEAS_AUG
GCM_SYSTEM_OUTFLOW
GCM_SYSTEM_THETA
/;


Model DRA_Access_Asymmetry_CostShare_Mkt
/
CSM_KKT_STAT_WD.WD
CSM_KKT_STAT_WS.WS
CSM_KKT_STAT_Q.Q
CSM_KKT_STAT_K.K
CSM_KKT_STAT_LR.LR
CSM_KKT_STAT_WP.WP_S
CSM_KKT_FEAS_LOSS.GMA_loss
CSM_KKT_FEAS_FLOW.GMA_flow
CSM_KKT_FEAS_CAP.GMA_cap
CSM_SYSTEM_MC.PI_as
CSM_KKT_FEAS_SUP
CSM_KKT_FEAS_AUG
CSM_SYSTEM_OUTFLOW
CSM_SYSTEM_THETA
/;

Model DRA_Access_Asymmetry_No_Mkt
/
NM_KKT_STAT_WD.WD
NM_KKT_STAT_WS.WS
NM_KKT_STAT_Q.Q
NM_KKT_STAT_K.K
NM_KKT_FEAS_FLOW.GMA_flow
NM_KKT_FEAS_CAP.GMA_cap
NM_KKT_FEAS_SUP
NM_KKT_FEAS_AUG
NM_SYSTEM_OUTFLOW
NM_SYSTEM_THETA
/;
"""
#Define general commodity market solve statement in GAMS syntax
def get_gcm_solve_text():
    return """
Parameters
model_stat_GCM General commodity market model solution status
net_benefit_GCM(i,t) General commodity market net benefits for player i in time period t
net_inflow_GCM(i,t) General commodity market net inflows for player i in time period t
min_inflow_GCM(i,t) General commodity market minimum inflows for player i in time period t
water_p_tot(i,t) General commodity market total water purchases for player i in time period t
min_dem_inf_GCM(i,t) General commodity market inflow at minimum demand level for player i in time period t
total_benefit_GCM General commodity market total basinwide benefits;

Solve DRA_Access_Asymmetry_Gen_Cmdty_Mkt using MCP;

model_stat_GCM = DRA_Access_Asymmetry_Gen_Cmdty_Mkt.modelStat;

net_benefit_GCM(i,pt) =
         round(
                 d(pt)*(alpha(i,pt)*Q.l(i,pt) - 0.5*beta*Q.l(i,pt)**2
                + PI_as.l(i,pt)*(sum(c,LR.l(i,c,pt))+WS.l(i,pt))
                - c_ops(i)*Q.l(i,pt) - sum(c, c_cu(i,c)*LR.l(i,c,pt))
                - c_sr(i)*WS.l(i,pt) - sum(j,delta_US_all(j,i)*PI_as.l(j,pt)*WP_G.l(i,j,pt))
                - c_cap(i)*K.l(i,pt))
         ,2)
                ;

total_benefit_GCM = round(sum((i,pt),net_benefit_GCM(i,pt)),2);

net_inflow_GCM(i,pt) = round(n(i) + O_min.l(i-1,pt) + sum((j,k_in),delta_US_all(j,i)*WP_G.l(k_in,j,pt)),2);

min_inflow_GCM(i,pt) = round(n(i) + O_min.l(i-1,pt),2);

water_p_tot(i,pt) = sum(j,delta_US_all(j,i)*WP_G.l(i,j,pt));

min_dem_inf_GCM(i,pt) = rfc(i)+Q.l(i,pt)-water_p_tot(i,pt);

Display net_benefit_GCM,total_benefit_GCM,net_inflow_GCM;
"""

def get_csm_solve_text():
    return """
Parameters
model_stat_CSM Cost sharing market model solution status
net_benefit_CSM(i,t) Cost sharing market net benefits for player i in time period t
net_inflow_CSM(i,t) Cost sharing market net inflows for player i in time period t
min_inflow_CSM(i,t) Cost sharing market minimum inflows for player i in time period t
min_dem_inf_CSM(i,t) Cost sharing market inflow at minimum demand level for player i in time period t
total_benefit_CSM Cost sharing market total basinwide benefits;

Solve DRA_Access_Asymmetry_CostShare_Mkt using MCP;

model_stat_CSM = DRA_Access_Asymmetry_CostShare_Mkt.modelStat;

net_benefit_CSM(i,pt) =
           round(
                 d(pt)*(alpha(i,pt)*Q.l(i,pt) - 0.5*beta*Q.l(i,pt)**2
                + sum(k_in, delta_DS_all(k_in,i)*PI_as.l(k_in,pt)*(
                    sum(c,LR.l(i,c,pt))+WS.l(i,pt))
                  )
                - c_ops(i)*Q.l(i,pt) - sum(c, c_cu(i,c)*LR.l(i,c,pt))
                - c_sr(i)*WS.l(i,pt) - PI_as.l(i,pt)*WP_S.l(i,pt)
                - c_cap(i)*K.l(i,pt))
                ,2
           )
                ;

total_benefit_CSM = round(sum((i,pt),net_benefit_CSM(i,pt)),2);

net_inflow_CSM(i,pt) = round(n(i) + O_min.l(i-1,pt) + sum(j, delta_US_all(j,i)*(sum(c,LR.l(j,c,pt))+WS.l(j,pt))),2);
*net_inflow_CSM(i,pt) = round(n(i) + O_min.l(i-1,pt) + WP_S.l(i,pt),2);

min_inflow_CSM(i,pt) = round(n(i) + O_min.l(i-1,pt),2);

min_dem_inf_CSM(i,pt) = rfc(i)+Q.l(i,pt)-WP_S.l(i,pt);

Display net_benefit_CSM,total_benefit_CSM,net_inflow_CSM;
"""

def get_nm_solve_text():
    return """Parameters
model_stat_NM No market model solution status
net_benefit_NM(i,t) No market net benefits for player i in time period t
net_inflow_NM(i,t) No market net inflows for player i in time period t
min_dem_inf_NM(i,t) No market inflow at minimum demand level for player i in time period t
total_benefit_NM No market total basinwide benefits
;

Solve DRA_Access_Asymmetry_No_Mkt using MCP;

model_stat_NM = DRA_Access_Asymmetry_No_Mkt.modelStat;

net_benefit_NM(i,pt) =
         round(
                 d(pt)*(alpha(i,pt)*Q.l(i,pt) - 0.5*beta*Q.l(i,pt)**2
               - c_ops(i)*Q.l(i,pt) - c_sr(i)*WS.l(i,pt)
                - c_cap(i)*K.l(i,pt)) ,2)
                ;

total_benefit_NM = round(sum((i,pt),net_benefit_NM(i,pt)),2);

net_inflow_NM(i,pt) = round(n(i) + O_min.l(i-1,pt),2);

min_dem_inf_NM(i,pt) = round(rfc(i)+Q.l(i,pt),2);

Display net_benefit_NM,total_benefit_NM,net_inflow_NM;
"""

#Build GAMS database from python data structures
def build_db(ws,players,classes,time_periods,V,prd,cyc,ir,beta,n,rfc,c_ops,c_sr,c_cap,c_cu,a_req,demand,lf,srt_pts):
    db_out = ws.add_database()
        
    i = db_out.add_set("i", 1, "players")
    for p in players:
        i.add_record(p)
            
    c = db_out.add_set("c", 1, "consumptive use classes")
    for cl in classes:
        c.add_record(cl)
            
    t = db_out.add_set("t", 1, "time periods")
    for tp in time_periods:
        t.add_record(tp)

    var_st = db_out.add_set("v", 1, "specified variable starting points")
    for v in V:
        var_st.add_record(v)
    
    cycles = gams.control.database.GamsParameter(db_out, "cyc", 0, "planning cycles")
    cycles.add_record().value = cyc
        
    interest_rate = gams.control.database.GamsParameter(db_out, "ir", 0, "interest rate")
    interest_rate.add_record().value = ir

    period = gams.control.database.GamsParameter(db_out, "prd", 0, "period of interval between time units")
    period.add_record().value = prd

    b = gams.control.database.GamsParameter(db_out, "beta",0,"inverse demand slope")
    b.add_record().value = beta

    inflow = gams.control.database.GamsParameter(db_out, "n", 1, "inflow for each player")
    for p in players:
        inflow.add_record(p).value = n[p]

    reg_flow = gams.control.database.GamsParameter(db_out, "rfc", 1, "regulatory flow constraint")
    for p in players:
        reg_flow.add_record(p).value = rfc[p]

    cost_ops = gams.control.database.GamsParameter(db_out, "c_ops", 1, "operating cost")
    for p in players:
        cost_ops.add_record(p).value = c_ops[p]

    cost_sr = gams.control.database.GamsParameter(db_out, "c_sr", 1, "storage costs")
    for p in players:
        cost_sr.add_record(p).value = c_sr[p]

    cost_cap = gams.control.database.GamsParameter(db_out, "c_cap", 1, "capital costs")
    for p in players:
        cost_cap.add_record(p).value = c_cap[p]

    cost_cu = gams.control.database.GamsParameter(db_out, "c_cu", 2, "consumptive use costs")
    for p in players:
        for cl in classes:
            cost_cu.add_record((p,cl)).value = c_cu[p,cl]

    augment_reqd = gams.control.database.GamsParameter(db_out, "a_req", 2, "capital augmentation required")
    for p in players:
        for tp in time_periods:
            augment_reqd.add_record((p,tp)).value = a_req[p,tp]

    water_demands = gams.control.database.GamsParameter(db_out, "demand", 2, "water demands")
    for p in players:
        for tp in time_periods:
            water_demands.add_record((p,tp)).value = demand[p,tp]

    loss_fract = gams.control.database.GamsParameter(db_out, "lf", 3, "loss fractions")
    for cl in classes:
        for p in players:
            for tp in time_periods:
                loss_fract.add_record((cl,p,tp)).value = lf[cl,p,tp]

    starting_points = gams.control.database.GamsParameter(db_out, "srt_pts", 1, "starting points for solver")
    for v in V:
        starting_points.add_record(v).value = srt_pts[v]
    
    db_out.export("model_data.gdx")

class GamsModel:
    #Build model
    def __init__(self,wd,players,classes,time_periods,V,prd,cyc,ir,beta,n,rfc,c_ops,c_sr,c_cap,c_cu,a_req,demand,lf,srt_pts):
        self.wd = wd
        self.players = players
        self.classes = classes
        self.time_periods = time_periods
        self.V = V
        self.prd = prd
        self.cyc = cyc
        self.ir = ir
        self.beta = beta
        self.n = n
        self.rfc = rfc
        self.c_ops = c_ops
        self.c_sr = c_sr
        self.c_cap = c_cap
        self.c_cu = c_cu
        self.demand = demand
        self.a_req = a_req
        self.lf = lf
        self.srt_pts = srt_pts
        self.ws = gams.control.workspace.GamsWorkspace(self.wd,system_directory="C:\\GAMS\\" + '43')
        self.gcm_total_text = get_model_text()+get_gcm_solve_text()
        self.csm_total_text = get_model_text()+get_csm_solve_text()
        self.nm_total_text = get_model_text()+get_nm_solve_text()

        build_db(self.ws,players,classes,time_periods,V,prd,cyc,ir,beta,n,rfc,c_ops,c_sr,c_cap,c_cu,a_req,demand,lf,srt_pts)


    def run_gcm_model(self):
        #Run general commodity market model
        rbe_gcm = self.ws.add_job_from_string(self.gcm_total_text)
        opt = self.ws.add_options()
        opt.defines["gdxincname"] = "model_data"
        opt.limrow = 10000
        opt.limcol = 10000
        rbe_gcm.run(opt)
        #Get outputs from general commodity market model run
        #Parameter outputs
        gcm_solve_stat = rbe_gcm.out_db["model_stat_GCM"].first_record().value
        
        net_benefit_gcm = {}
        for rec in rbe_gcm.out_db["net_benefit_GCM"]:
            net_benefit_gcm["net_benefit_gcm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        total_benefit_gcm = rbe_gcm.out_db["total_benefit_GCM"].first_record().value
        
        net_inflow_gcm = {}
        for rec in rbe_gcm.out_db["net_inflow_GCM"]:
            net_inflow_gcm["net_inflow_gcm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        min_inflow_gcm = {}
        for rec in rbe_gcm.out_db["min_inflow_GCM"]:
            min_inflow_gcm["min_inflow_gcm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        water_p_tot = {}
        for rec in rbe_gcm.out_db["water_p_tot"]:
            water_p_tot[tuple(rec.keys)] = rec.value
        wp_sparse = {} # Loop to add zeros to players with no water purchases
        for i,t in itertools.product(self.players,self.time_periods):
            if (i,t) in water_p_tot.keys():
                wp_sparse["wp_sparse"+'%s'%str(tuple((i,t)))] = water_p_tot[i,t]
            else:
                wp_sparse["wp_sparse"+'%s'%str(tuple((i,t)))] = 0
                
        min_dem_inf_gcm = {}
        for rec in rbe_gcm.out_db["min_dem_inf_GCM"]:
            min_dem_inf_gcm["min_dem_inf_gcm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        #Variable outputs
        gcm_var_dict = {}
        gams_var = ["WD","WS","Q","K","LR","WP_G","GMA_loss","GMA_flow","GMA_cap","PI_as","O_min","LMB_sup","LMB_aug","THT"]
        for var in gams_var:
            for rec in rbe_gcm.out_db[var]:
                gcm_var_dict[var+'%s'%str(tuple(rec.keys))] = round(rec.level,2)
                #gcm_var_dict[var+'%s'%str(tuple(rec.keys))] = rec.level
        return gcm_solve_stat, net_benefit_gcm, total_benefit_gcm, net_inflow_gcm, min_inflow_gcm, wp_sparse, min_dem_inf_gcm, gcm_var_dict


    def run_csm_model(self):
        #Run cost sharing market model
        rbe_csm = self.ws.add_job_from_string(self.csm_total_text)
        opt = self.ws.add_options()
        opt.defines["gdxincname"] = "model_data"
        opt.limrow = 10000
        opt.limcol = 10000
        rbe_csm.run(opt)
        #Get outputs from cost sharing market model run
        #Parameter outputs
        csm_solve_stat = rbe_csm.out_db["model_stat_CSM"].first_record().value

        net_benefit_csm = {}
        for rec in rbe_csm.out_db["net_benefit_CSM"]:
            net_benefit_csm["net_benefit_csm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        total_benefit_csm = rbe_csm.out_db["total_benefit_CSM"].first_record().value
        
        net_inflow_csm = {}
        for rec in rbe_csm.out_db["net_inflow_CSM"]:
            net_inflow_csm["net_inflow_csm"+'%s'%str(tuple(rec.keys))] = rec.value

        min_inflow_csm = {}
        for rec in rbe_csm.out_db["min_inflow_CSM"]:
            min_inflow_csm["min_inflow_csm"+'%s'%str(tuple(rec.keys))] = rec.value
             
        min_dem_inf_csm = {}
        for rec in rbe_csm.out_db["min_dem_inf_CSM"]:
            min_dem_inf_csm["min_dem_inf_csm"+'%s'%str(tuple(rec.keys))] = rec.value
        
        #Variable outputs
        csm_var_dict = {}
        gams_var = ["WD","WS","Q","K","LR","WP_S","GMA_loss","GMA_flow","GMA_cap","PI_as","O_min","LMB_sup","LMB_aug","THT"]
        for var in gams_var:
            for rec in rbe_csm.out_db[var]:
                csm_var_dict[var+'%s'%str(tuple(rec.keys))] = round(rec.level,2)
                #csm_var_dict[var+'%s'%str(tuple(rec.keys))] = rec.level

        return csm_solve_stat, net_benefit_csm, total_benefit_csm, net_inflow_csm, min_inflow_csm, min_dem_inf_csm, csm_var_dict


    def run_nm_model(self):
        #Run no market model
        rbe_nm = self.ws.add_job_from_string(self.nm_total_text)
        opt = self.ws.add_options()
        opt.defines["gdxincname"] = "model_data"
        opt.limrow = 10000
        opt.limcol = 10000
        rbe_nm.run(opt)
        #Get outputs from no market model run
        #Parameter outputs
        nm_solve_stat = rbe_nm.out_db["model_stat_NM"].first_record().value

        net_benefit_nm = {}
        for rec in rbe_nm.out_db["net_benefit_NM"]:
            net_benefit_nm["net_benefit_nm"+'%s'%str(tuple(rec.keys))] = rec.value

        total_benefit_nm = rbe_nm.out_db["total_benefit_NM"].first_record().value

        net_inflow_nm = {}
        for rec in rbe_nm.out_db["net_inflow_NM"]:
            net_inflow_nm["net_inflow_nm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        min_dem_inf_nm = {}
        for rec in rbe_nm.out_db["min_dem_inf_NM"]:
            min_dem_inf_nm["min_dem_inf_nm"+'%s'%str(tuple(rec.keys))] = rec.value
            
        
        #Variable outputs
        nm_var_dict = {}
        gams_var = ["WD","WS","Q","K","GMA_flow","GMA_cap","O_min","LMB_sup","LMB_aug","THT"]
        for var in gams_var:
            for rec in rbe_nm.out_db[var]:
                nm_var_dict[var+'%s'%str(tuple(rec.keys))] = round(rec.level,2)
                #nm_var_dict[var+'%s'%str(tuple(rec.keys))] = rec.level

        return nm_solve_stat, net_benefit_nm, total_benefit_nm, net_inflow_nm, min_dem_inf_nm, nm_var_dict

        

    
        

    
