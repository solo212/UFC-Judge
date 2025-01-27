# UFC-Judge
Upload a CSV of given format and see what fighter will win


csv has to be given these metrics: 

B_KD: Number of knockdowns by the blue fighter.
R_SIG_STR_pct: Significant strike accuracy (percentage) by the red fighter (value between 0 and 1).
B_SIG_STR_pct: Significant strike accuracy (percentage) by the blue fighter (value between 0 and 1).
R_TD_pct: Takedown accuracy (percentage) by the red fighter (value between 0 and 1).
B_TD_pct: Takedown accuracy (percentage) by the blue fighter (value between 0 and 1).
R_SUB_ATT: Number of submission attempts by the red fighter.
B_SUB_ATT: Number of submission attempts by the blue fighter.
R_REV: Number of reversals by the red fighter.
B_REV: Number of reversals by the blue fighter.
last_round: The final round number of the fight (e.g., 3 for a 3-round fight, 5 for a championship fight).
R_SIG_STR_Landed: Total significant strikes landed by the red fighter.
R_SIG_STR_Attempted: Total significant strikes attempted by the red fighter.
B_SIG_STR_Landed: Total significant strikes landed by the blue fighter.
B_SIG_STR_Attempted: Total significant strikes attempted by the blue fighter.
R_Strike_Efficiency: Red fighter’s strike efficiency (calculated as landed strikes / attempted strikes).
B_Strike_Efficiency: Blue fighter’s strike efficiency (calculated as landed strikes / attempted strikes).
R_Control_Time: Total control time by the red fighter (in seconds).
B_Control_Time: Total control time by the blue fighter (in seconds).
Control_Time_Difference: Difference in control time between red and blue fighters (R_Control_Time - B_Control_Time).
R_TOTAL_STR._Landed: Total strikes landed by the red fighter.
R_TOTAL_STR._Attempted: Total strikes attempted by the red fighter.
B_TOTAL_STR._Landed: Total strikes landed by the blue fighter.
B_TOTAL_STR._Attempted: Total strikes attempted by the blue fighter.
R_TD_Landed: Total takedowns landed by the red fighter.
R_TD_Attempted: Total takedowns attempted by the red fighter.
B_TD_Landed: Total takedowns landed by the blue fighter.
B_TD_Attempted: Total takedowns attempted by the blue fighter.
R_HEAD_Landed: Total head strikes landed by the red fighter.
R_HEAD_Attempted: Total head strikes attempted by the red fighter.
B_HEAD_Landed: Total head strikes landed by the blue fighter.
B_HEAD_Attempted: Total head strikes attempted by the blue fighter.
R_BODY_Landed: Total body strikes landed by the red fighter.
R_BODY_Attempted: Total body strikes attempted by the red fighter.
B_BODY_Landed: Total body strikes landed by the blue fighter.
B_BODY_Attempted: Total body strikes attempted by the blue fighter.
R_LEG_Landed: Total leg strikes landed by the red fighter.
R_LEG_Attempted: Total leg strikes attempted by the red fighter.
B_LEG_Landed: Total leg strikes landed by the blue fighter.
B_LEG_Attempted: Total leg strikes attempted by the blue fighter.
R_DISTANCE_Landed: Total distance strikes landed by the red fighter.
R_DISTANCE_Attempted: Total distance strikes attempted by the red fighter.
B_DISTANCE_Landed: Total distance strikes landed by the blue fighter.
B_DISTANCE_Attempted: Total distance strikes attempted by the blue fighter.
R_CLINCH_Landed: Total clinch strikes landed by the red fighter.
R_CLINCH_Attempted: Total clinch strikes attempted by the red fighter.
B_CLINCH_Landed: Total clinch strikes landed by the blue fighter.
B_CLINCH_Attempted: Total clinch strikes attempted by the blue fighter.
R_GROUND_Landed: Total ground strikes landed by the red fighter.
R_GROUND_Attempted: Total ground strikes attempted by the red fighter.
B_GROUND_Landed: Total ground strikes landed by the blue fighter.
B_GROUND_Attempted: Total ground strikes attempted by the blue fighter.
last_round_time_seconds: Time remaining in the last round of the fight (in seconds).
R_CTRL_seconds: Total control time for the red fighter (in seconds).
B_CTRL_seconds: Total control time for the blue fighter (in seconds).

results are titled rf_predictions and gb_predictions: 0 means red fighter wins, 1 means blue fighter wins for both
