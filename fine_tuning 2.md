Fine-tuning with the Gemini API
 

Prompt design strategies such as few-shot prompting may not always produce the results you need. Fine-tuning is a process that can improve your model's performance on specific tasks or help the model adhere to specific output requirements when instructions aren't sufficient and you have a set of examples that demonstrate the outputs you want.

This page provides a conceptual overview of fine-tuning the text model behind the Gemini API text service. When you're ready to start tuning, try the fine-tuning tutorial. If you'd like a more general introduction to customizing LLMs for specific use cases, check out LLMs: Fine-tuning, distillation, and prompt engineering in the Machine Learning Crash Course.

How fine-tuning works

The goal of fine-tuning is to further improve the performance of the model for your specific task. Fine-tuning works by providing the model with a training dataset containing many examples of the task. For niche tasks, you can get significant improvements in model performance by tuning the model on a modest number of examples. This kind of model tuning is sometimes referred to as supervised fine-tuning, to distinguish it from other kinds of fine-tuning.

Your training data should be structured as examples with prompt inputs and expected response outputs. You can also tune models using example data directly in Google AI Studio. The goal is to teach the model to mimic the wanted behavior or task, by giving it many examples illustrating that behavior or task.

When you run a tuning job, the model learns additional parameters that help it encode the necessary information to perform the wanted task or learn the wanted behavior. These parameters can then be used at inference time. The output of the tuning job is a new model, which is effectively a combination of the newly learned parameters and the original model.

Prepare your dataset

Before you can start fine-tuning, you need a dataset to tune the model with. For the best performance, the examples in the dataset should be of high quality, diverse, and representative of real inputs and outputs.

Format

Note: Fine-tuning only supports input-output pair examples. Chat-style multi-turn conversations are not supported at this time.
The examples included in your dataset should match your expected production traffic. If your dataset contains specific formatting, keywords, instructions, or information, the production data should be formatted in the same way and contain the same instructions.

For example, if the examples in your dataset include a "question:" and a "context:", production traffic should also be formatted to include a "question:" and a "context:" in the same order as it appears in the dataset examples. If you exclude the context, the model can't recognize the pattern, even if the exact question was in an example in the dataset.

As another example, here's Python training data for an application that generates the next number in a sequence:


training_data = [
  {"text_input": "1", "output": "2"},
  {"text_input": "3", "output": "4"},
  {"text_input": "-3", "output": "-2"},
  {"text_input": "twenty two", "output": "twenty three"},
  {"text_input": "two hundred", "output": "two hundred one"},
  {"text_input": "ninety nine", "output": "one hundred"},
  {"text_input": "8", "output": "9"},
  {"text_input": "-98", "output": "-97"},
  {"text_input": "1,000", "output": "1,001"},
  {"text_input": "10,100,000", "output": "10,100,001"},
  {"text_input": "thirteen", "output": "fourteen"},
  {"text_input": "eighty", "output": "eighty one"},
  {"text_input": "one", "output": "two"},
  {"text_input": "three", "output": "four"},
  {"text_input": "seven", "output": "eight"},
]
Adding a prompt or preamble to each example in your dataset can also help improve the performance of the tuned model. Note, if a prompt or preamble is included in your dataset, it should also be included in the prompt to the tuned model at inference time.

Limitations

Note: Fine-tuning datasets for Gemini 1.5 Flash have the following limitations:

The maximum input size per example is 40,000 characters.
The maximum output size per example is 5,000 characters.
Training data size

You can fine-tune a model with as little as 20 examples. Additional data generally improves the quality of the responses. You should target between 100 and 500 examples, depending on your application. The following table shows recommended dataset sizes for fine-tuning a text model for various common tasks:

Task	No. of examples in dataset
Classification	100+
Summarization	100-500+
Document search	100+
Upload your tuning dataset

Data is either passed inline using the API or through files uploaded in Google AI Studio.

To use the client library, provide the data file in the createTunedModel call. File size limit is 4 MB. See the fine-tuning quickstart with Python to get started.

To call the REST API using cURL, provide training examples in JSON format to the training_data argument. See the tuning quickstart with cURL to get started.

Advanced tuning settings

When creating a tuning job, you can specify the following advanced settings:

Epochs: A full training pass over the entire training set such that each example has been processed once.
Batch size: The set of examples used in one training iteration. The batch size determines the number of examples in a batch.
Learning rate: A floating-point number that tells the algorithm how strongly to adjust the model parameters on each iteration. For example, a learning rate of 0.3 would adjust weights and biases three times more powerfully than a learning rate of 0.1. High and low learning rates have their own unique trade-offs and should be adjusted based on your use case.
Learning rate multiplier: The rate multiplier modifies the model's original learning rate. A value of 1 uses the original learning rate of the model. Values greater than 1 increase the learning rate and values between 1 and 0 lower the learning rate.
Recommended configurations

The following table shows the recommended configurations for fine-tuning a foundation model:

Hyperparameter	Default value	Recommended adjustments
Epoch	5	
If the loss starts to plateau before 5 epochs, use a smaller value.

If the loss is converging and doesn't seem to plateau, use a higher value.
Batch size	4	
Learning rate	0.001	Use a smaller value for smaller datasets.
The loss curve shows how much the model's prediction deviates from the ideal predictions in the training examples after each epoch. Ideally you want to stop training at the lowest point in the curve right before it plateaus. For example, the graph below shows the loss curve plateauing at about epoch 4-6 which means you can set the Epoch parameter to 4 and still get the same performance.

Line chart showing the loss curve for the model. The line spikes between the
first and the second epochs, then sharply declines to almost 0 and levels out
after three epochs.

Check the tuning job status

You can check the status of your tuning job in Google AI Studio under the My Library tab or using the metadata property of the tuned model in the Gemini API.

Troubleshoot errors

This section includes tips on how to resolve errors you may encounter while creating your tuned model.

Authentication

Note: Starting September 30, 2024, OAuth authentication is no longer required. New projects should use API key authentication instead.
Tuning using the API and client library requires authentication. You can set up authentication using either an API key (recommended) or using OAuth credentials. For documentation on setting up an API key, see Set up API key.

If you see a 'PermissionDenied: 403 Request had insufficient authentication scopes' error, you may need to set up user authentication using OAuth credentials. To configure OAuth credentials for Python, visit our the OAuth setup tutorial.

Canceled models

You can cancel a fine-tuning job any time before the job is finished. However, the inference performance of a canceled model is unpredictable, particularly if the tuning job is canceled early in the training. If you canceled because you want to stop the training at an earlier epoch, you should create a new tuning job and set the epoch to a lower value.

Limitations of tuned models

Note: Tuned models have the following limitations:

The input limit of a tuned Gemini 1.5 Flash model is 40,000 characters.
JSON mode is not supported with tuned models.
Only text input is supported.
What's next

Get started with the fine-tuning tutorials:

Fine-tuning tutorial (Python)
Fine-tuning tutorial (REST)
Fine-tuning tutorial
 

Python  REST

This tutorial will help you get started with the Gemini API tuning service using either the Python SDK or the REST API using curl. The examples show how to tune the text model behind the Gemini API text generation service.

Try a Colab notebook
View notebook on GitHub
Limitations

Before tuning a model, you should be aware of the following limitations:

Fine-tuning datasets

Fine-tuning datasets for Gemini 1.5 Flash have the following limitations:

The maximum input size per example is 40,000 characters.
The maximum output size per example is 5,000 characters.
Only input-output pair examples are supported. Chat-style multi-turn conversations are not supported.
Tuned models

Tuned models have the following limitations:

The input limit of a tuned Gemini 1.5 Flash model is 40,000 characters.
JSON mode is not supported with tuned models.
Only text input is supported.
Before you begin: Set up your project and API key

Before calling the Gemini API, you need to set up your project and configure your API key.

 Expand to view how to set up your project and API key

List tuned models

You can check your existing tuned models with the tunedModels.list method.


from google import genai
from google.genai import types
client = genai.Client() # Get the key from the GOOGLE_API_KEY env variable

for model_info in client.models.list():
    print(model_info.name)
Create a tuned model

To create a tuned model, you need to pass your dataset to the model in the tunedModels.create method.

For this example, you will tune a model to generate the next number in the sequence. For example, if the input is 1, the model should output 2. If the input is one hundred, the output should be one hundred one.


# create tuning model
training_dataset =  [
    ["1", "2"],
    ["3", "4"],
    ["-3", "-2"],
    ["twenty two", "twenty three"],
    ["two hundred", "two hundred one"],
    ["ninety nine", "one hundred"],
    ["8", "9"],
    ["-98", "-97"],
    ["1,000", "1,001"],
    ["10,100,000", "10,100,001"],
    ["thirteen", "fourteen"],
    ["eighty", "eighty one"],
    ["one", "two"],
    ["three", "four"],
    ["seven", "eight"],
]
training_dataset=types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=i,
                output=o,
            )
            for i,o in training_dataset
        ],
    )
tuning_job = client.tunings.tune(
    base_model='models/gemini-1.5-flash-001-tuning',
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count= 5,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name="test tuned model"
    )
)

# generate content with the tuned model
response = client.models.generate_content(
    model=tuning_job.tuned_model.model,
    contents='III',
)

print(response.text)
The optimal values for epoch count, batch size, and learning rate are dependent on your dataset and other constraints of your use case. To learn more about these values, see Advanced tuning settings and Hyperparameters.

Tip: For a more general introduction to these hyperparameters, see Linear regression: Hyperparameters in the Machine Learning Crash Course.
Try the model

You can use the tunedModels.generateContent method and specify the name of the tuned model to test its performance.


response = client.models.generate_content(
    model=tuning_job.tuned_model.model,
    contents='III'
)
Not implemented

Some features (progress reporting, updating the description, and deleting tuned models) has not yet been implemented in the new SDK.
Send feedback Fine-tuning tutorial
 

Python  REST

This tutorial will help you get started with the Gemini API tuning service using either the Python SDK or the REST API using curl. The examples show how to tune the text model behind the Gemini API text generation service.

Try a Colab notebook
View notebook on GitHub
Limitations

Before tuning a model, you should be aware of the following limitations:

Fine-tuning datasets

Fine-tuning datasets for Gemini 1.5 Flash have the following limitations:

The maximum input size per example is 40,000 characters.
The maximum output size per example is 5,000 characters.
Only input-output pair examples are supported. Chat-style multi-turn conversations are not supported.
Tuned models

Tuned models have the following limitations:

The input limit of a tuned Gemini 1.5 Flash model is 40,000 characters.
JSON mode is not supported with tuned models.
Only text input is supported.
Before you begin: Set up your project and API key

Before calling the Gemini API, you need to set up your project and configure your API key.

 Expand to view how to set up your project and API key

List tuned models

You can check your existing tuned models with the tunedModels.list method.


# Sending a page_size is optional
curl -X GET https://generativelanguage.googleapis.com/v1beta/tunedModels?page_size=5 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${access_token}" \
    -H "x-goog-user-project: ${project_id}" > tuned_models.json

jq .tunedModels[].name < tuned_models.json

# Send the nextPageToken to get the next page.
page_token=$(jq .nextPageToken < tuned_models.json | tr -d '"')

if [[ "$page_token" != "null"" ]]; then
curl -X GET https://generativelanguage.googleapis.com/v1beta/tunedModels?page_size=5\&page_token=${page_token}?key=$GEMINI_API_KEY \
    -H "Content-Type: application/json"  > tuned_models2.json
jq .tunedModels[].name < tuned_models.json
fi
Create a tuned model

To create a tuned model, you need to pass your dataset to the model in the tunedModels.create method.

For this example, you will tune a model to generate the next number in the sequence. For example, if the input is 1, the model should output 2. If the input is one hundred, the output should be one hundred one.


curl -X POST "https://generativelanguage.googleapis.com/v1beta/tunedModels?key=$GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '
      {
        "display_name": "number generator model",
        "base_model": "models/gemini-1.5-flash-001-tuning",
        "tuning_task": {
          "hyperparameters": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "epoch_count":5,
          },
          "training_data": {
            "examples": {
              "examples": [
                {
                    "text_input": "1",
                    "output": "2",
                },{
                    "text_input": "3",
                    "output": "4",
                },{
                    "text_input": "-3",
                    "output": "-2",
                },{
                    "text_input": "twenty two",
                    "output": "twenty three",
                },{
                    "text_input": "two hundred",
                    "output": "two hundred one",
                },{
                    "text_input": "ninety nine",
                    "output": "one hundred",
                },{
                    "text_input": "8",
                    "output": "9",
                },{
                    "text_input": "-98",
                    "output": "-97",
                },{
                    "text_input": "1,000",
                    "output": "1,001",
                },{
                    "text_input": "10,100,000",
                    "output": "10,100,001",
                },{
                    "text_input": "thirteen",
                    "output": "fourteen",
                },{
                    "text_input": "eighty",
                    "output": "eighty one",
                },{
                    "text_input": "one",
                    "output": "two",
                },{
                    "text_input": "three",
                    "output": "four",
                },{
                    "text_input": "seven",
                    "output": "eight",
                }
              ]
            }
          }
        }
      }' | tee tunemodel.json

# Check the operation for status updates during training.
# Note: you can only check the operation on v1/
operation=$(cat tunemodel.json | jq ".name" | tr -d '"')
tuning_done=false

while [[ "$tuning_done" != "true" ]];
do
  sleep 5
  curl -X GET "https://generativelanguage.googleapis.com/v1/${operation}?key=$GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
     2> /dev/null > tuning_operation.json

  complete=$(jq .metadata.completedPercent < tuning_operation.json)
  tput cuu1
  tput el
  echo "Tuning...${complete}%"
  tuning_done=$(jq .done < tuning_operation.json)
done

# Or get the TunedModel and check it's state. The model is ready to use if the state is active.
modelname=$(cat tunemodel.json | jq ".metadata.tunedModel" | tr -d '"')
curl -X GET  https://generativelanguage.googleapis.com/v1beta/${modelname}?key=$GEMINI_API_KEY \
    -H 'Content-Type: application/json' > tuned_model.json

cat tuned_model.json | jq ".state"
The optimal values for epoch count, batch size, and learning rate are dependent on your dataset and other constraints of your use case. To learn more about these values, see Advanced tuning settings and Hyperparameters.

Tip: For a more general introduction to these hyperparameters, see Linear regression: Hyperparameters in the Machine Learning Crash Course.
Your tuned model is immediately added to the list of tuned models, but its state is set to "creating" while the model is tuned.

Try the model

You can use the tunedModels.generateContent method and specify the name of the tuned model to test its performance.


curl -X POST https://generativelanguage.googleapis.com/v1beta/$modelname:generateContent?key=$GEMINI_API_KEY \
    -H 'Content-Type: application/json' \
    -d '{
        "contents": [{
        "parts": [{
          "text": "LXIII"
          }]
        }]
        }' 2> /dev/null
Delete the model

You can clean up your tuned model list by deleting models you no longer need. Use the tunedModels.delete method to delete a model. If you canceled any tuning jobs, you may want to delete those as their performance may be unpredictable.


curl -X DELETE https://generativelanguage.googleapis.com/v1beta/${modelname}?key=$GEMINI_API_KEY \
    -H 'Content-Type: application/json' 
