import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(np.hstack((x, y)), ["x", "y"])

regression_model = LinearRegression(featuresCol="x", labelCol="y")
model = regression_model.fit(df)
predictions = model.transform(df)

evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)

print('The coefficient is {}'.format(model.coefficients))
print('The intercept is {}'.format(model.intercept))
print('Root mean squared error of the model is {}.'.format(rmse))
print('R-squared score is {}.'.format(r2))

x_values = [row[0] for row in df.select("x").collect()]
y_values = [row[0] for row in df.select("y").collect()]
y_predicted = [row[0] for row in predictions.select("prediction").collect()]

plt.scatter(x_values, y_values, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_values, y_predicted, color='r')
plt.show()

 
 
