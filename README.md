# UFC-Judge
Upload a CSV of given format and see what fighter will win


csv has to be given these metrics: 

1. B_KD: Number of knockdowns by the blue fighter.
2. R_SIG_STR_pct: Significant strike accuracy (percentage) by the red fighter (value between 0 and 1).
3.B_SIG_STR_pct: Significant strike accuracy (percentage) by the blue fighter (value between 0 and 1).
4.R_TD_pct: Takedown accuracy (percentage) by the red fighter (value between 0 and 1).
5.B_TD_pct: Takedown accuracy (percentage) by the blue fighter (value between 0 and 1).
6.R_SUB_ATT: Number of submission attempts by the red fighter.
7.B_SUB_ATT: Number of submission attempts by the blue fighter.
8.R_REV: Number of reversals by the red fighter.
9.B_REV: Number of reversals by the blue fighter.
10.last_round: The final round number of the fight (e.g., 3 for a 3-round fight, 5 for a championship fight).
11.R_SIG_STR_Landed: Total significant strikes landed by the red fighter.
12.R_SIG_STR_Attempted: Total significant strikes attempted by the red fighter.
13.B_SIG_STR_Landed: Total significant strikes landed by the blue fighter.
14.B_SIG_STR_Attempted: Total significant strikes attempted by the blue fighter.
15.R_Strike_Efficiency: Red fighter’s strike efficiency (calculated as landed strikes / attempted strikes).
16.B_Strike_Efficiency: Blue fighter’s strike efficiency (calculated as landed strikes / attempted strikes).
17.R_Control_Time: Total control time by the red fighter (in seconds).
18.B_Control_Time: Total control time by the blue fighter (in seconds).
19.Control_Time_Difference: Difference in control time between red and blue fighters (R_Control_Time - B_Control_Time).
20.R_TOTAL_STR._Landed: Total strikes landed by the red fighter.
21.R_TOTAL_STR._Attempted: Total strikes attempted by the red fighter.
22.B_TOTAL_STR._Landed: Total strikes landed by the blue fighter.
23.B_TOTAL_STR._Attempted: Total strikes attempted by the blue fighter.
24.R_TD_Landed: Total takedowns landed by the red fighter.
25.R_TD_Attempted: Total takedowns attempted by the red fighter.
26.B_TD_Landed: Total takedowns landed by the blue fighter.
27.B_TD_Attempted: Total takedowns attempted by the blue fighter.
28.R_HEAD_Landed: Total head strikes landed by the red fighter.
29.R_HEAD_Attempted: Total head strikes attempted by the red fighter.
30.B_HEAD_Landed: Total head strikes landed by the blue fighter.
31.B_HEAD_Attempted: Total head strikes attempted by the blue fighter.
32.R_BODY_Landed: Total body strikes landed by the red fighter.
33.R_BODY_Attempted: Total body strikes attempted by the red fighter.
34.B_BODY_Landed: Total body strikes landed by the blue fighter.
35.B_BODY_Attempted: Total body strikes attempted by the blue fighter.
36.R_LEG_Landed: Total leg strikes landed by the red fighter.
37.R_LEG_Attempted: Total leg strikes attempted by the red fighter.
38.B_LEG_Landed: Total leg strikes landed by the blue fighter.
39.B_LEG_Attempted: Total leg strikes attempted by the blue fighter.
40.R_DISTANCE_Landed: Total distance strikes landed by the red fighter.
41.R_DISTANCE_Attempted: Total distance strikes attempted by the red fighter.
42.B_DISTANCE_Landed: Total distance strikes landed by the blue fighter.
43.B_DISTANCE_Attempted: Total distance strikes attempted by the blue fighter.
44.R_CLINCH_Landed: Total clinch strikes landed by the red fighter.
45.R_CLINCH_Attempted: Total clinch strikes attempted by the red fighter.
46.B_CLINCH_Landed: Total clinch strikes landed by the blue fighter.
47.B_CLINCH_Attempted: Total clinch strikes attempted by the blue fighter.
48.R_GROUND_Landed: Total ground strikes landed by the red fighter.
49.R_GROUND_Attempted: Total ground strikes attempted by the red fighter.
50.B_GROUND_Landed: Total ground strikes landed by the blue fighter.
51.B_GROUND_Attempted: Total ground strikes attempted by the blue fighter.
52.last_round_time_seconds: Time remaining in the last round of the fight (in seconds).
53.R_CTRL_seconds: Total control time for the red fighter (in seconds).
54.B_CTRL_seconds: Total control time for the blue fighter (in seconds).

results are titled rf_predictions and gb_predictions: 0 means red fighter wins, 1 means blue fighter wins for both
