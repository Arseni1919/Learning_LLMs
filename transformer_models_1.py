from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# output = classifier("I've been waiting for a HuggingFace course my whole life.")
#
# print(output)


generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

