# test functions implemented in entry_point.py and deploy_ei.py
import entry_point

print("=============== Test regular entry point =================")
# load model
model = entry_point.model_fn("")

# single sentence
request_body = '{"review_body": "Don\u2019t buy! I ordered Mango flavor specifically for flavor and received regular Fruit Punch! Waste of money and false advertisement!", "review_title": "Don\u2019t buy! Will not get the product you pay for"}'
data = entry_point.input_fn(request_body, "application/json")

output = entry_point.predict_fn(data, model)

print(output[0])
