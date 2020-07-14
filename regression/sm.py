import statsmodels.api as sm

# Load the data from Spector and Mazzeo (1980)
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)

# Follow statsmodles ipython notebook
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit(disp=0)

print('Parameters: ', logit_res.params)
print(logit_res.summary())
