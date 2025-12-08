Todos

# HIGH PRIORITY
- add hyperparameter tuning
- try modelling trend and see what happens. For example, use ARIMA on y:
	-	yt \approx g't:= gt + et$, then
	-	use predictions g't to compute residuals rt := yt - g't
	-	use ML to predict r

# MEDIUM PRIORITY
- add prophet, arima, sarima
- when adding SARIMAX, linear models, prophet or neural networks, consider correlation-based pruning as a preprocessing step


# LOW PRIORITY
- try on fast.api and PyDantic
- create endpoints with: a) everything until training -> save pipeline object; b) inference 
