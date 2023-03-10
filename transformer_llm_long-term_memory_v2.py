"""
This program can be used to generate long text strings through the 
openai.ChatCompletions API. For a given prompt, this program will generate:
(1) a version of the output that uses the original prompt, summarization of 
all preceding text, and a copy of the last generated paragraph as context for 
each inference/completion, and (2) a version of the output that only uses the 
original prompat and a copy of the last generated paragraph as context for each 
inference/completion. These outputs will be stored in two separate files in a 
directory of the user's choosing and with a filename the user also chooses. 
This program also generates log files associated with each output that show the 
response associated with each API call as well as the prompt/context used for the 
inference (including last paragraph and, if used, the summary of the preceding text).  

The purpose of this program is to test whether it is possible to leverage 
the summarization capabilities of transformer-based LLMs to give them a sort of 
"long-term memory" and enable them to generate better/more coherent long-form text. 

PLEASE NOTE: The 'directory,' 'file_name,'' and 'prompt' command line 
variables are all required. All other variables are optional and will be 
set to default values if the user doesn't specify values for them. To see
a list of all variables the user can control and descriptions of each of 
those variables, use the "-h" flag when running the program from the command line.

PLEASE ALSO NOTE: In order for this code to work, you need to create an environment 
variable called 'OPENAI_API_KEY' and store the value of your OpenAI API key there.   
"""


import argparse # import argparse module to enable user to pass labeled arguments to program through command line 
import numpy as np # import numpy 
import openai # import openai module to use for api calls
import tiktoken # import openai tokenizer to use to determine how many tokens are in a text string and to decode tokenized strings
import os # import os module to enable changing of working directory to wherever user wants to store responses and to enable retrieval of API key from environment variable


# The code below parses arguments provided by user. 
parser = argparse.ArgumentParser() #Create parser object

parser.add_argument("--max_output_tokens_generation", type=int, choices=np.asarray(np.linspace(100, 1000, 10), dtype = 'int'), default=500, help = "maximum number of output tokens for a generation API call") 
parser.add_argument("--max_output_tokens_summarization", type=int, choices=np.asarray(np.linspace(100, 1000, 10), dtype = 'int'), default=200, help = "maximum number of output tokens for a summarization API call") 
parser.add_argument("--temperature", type=float, default=0, help = "temperature for API calls. should be between 0 and 2.") 
parser.add_argument("--num_completions", type=int, choices=range(1,21), default=10, help="number of completions to get before finishing text generation")
parser.add_argument("--context_length", type=int, choices=np.asarray(np.linspace(100, 1000, 10), dtype = 'int'), default=500, help="how many tokens from last generation to include in context for next generation")
parser.add_argument("--directory", type=str, required = True, help = "filepath for directory where user wishes files containing generated text to be stored. the user must enter something for this variable.")
parser.add_argument("--file_name", type=str, required = True, help= "what name the user wishes files containing generated text to have. '_with_summarization.txt' or '_without_summarization.txt' will be apended to the end of this name. the user must enter something for this variable.")
parser.add_argument("--prompt", type=str, required = True, help = "prompt for first API call. the user must enter something for this variable.")
parser.add_argument("--additional_instruction", type=str, default="Please only write the first part/introduction of this for now. Someone should be able to naturally continue writing from where you left off.", help = "additional instructions to append to user-provided prompt for first API call")
parser.add_argument("--initial_summarization_prompt", type=str, default="Please summarize the text below in {} tokens or less.", help = "prompt used to ask API to summarize the first segment of text generated")
parser.add_argument("--continuation_prompt", type=str, default="You are writing a long sequence of text. The text immediately below contains the original prompt. The text below that contains a summary of everything you've written so far. The text below that contains a copy of some of the text at the end of the text sequence you've written so far. Please continue writing from where you left off, but do not finish the text. Someone should be able to naturally continue writing from where you left off.", help = "prompt used to ask API to generate more text, continuing from where it left off with the original prompt, the last paragraph, and a summary of all previous text as context. note that if you provide a prompt that specifies a different syntax from the default prompt, you should probably also change the syntax in the code that makes the api call to generate a continuation.")
parser.add_argument("--continuation_without_summarization_prompt", type=str, default="You are writing a long sequence of text. The text immediately below contains the original prompt. The text below that contains a copy of some of the text at the end of the text sequence you've written so far. Please continue writing from where you left off, but do not finish the text. Someone should be able to naturally continue writing from where you left off.", help = "prompt used to ask API to generate more text, with the original prompt and the last paragraph as context. note that if you provide a prompt that specifies a different syntax from the default prompt, you should probably also change the syntax in the code that makes the api call to generate a continuation.")
parser.add_argument("--continuation_summarization_prompt", type=str, default="You are writing a long sequence of text. The text immediately below contains the original prompt. The text below that contains a summary of everything you've written except some of the text at the end of what you've written. The text below that contains a copy of the text you've written that hasn't yet been incorporated into the summary. Please write a new summary of what you've written so far in {} tokens or less. The new summary should incorporate the most important information from both the previous summary and the text that hasn't yet been incorporated into the summary. Please make sure the new summary you write ends in a complete sentence and not a sentence that it is cut off.", help = "prompt used to create a new summary incorporating information from recently generated paragraphs.")
parser.add_argument("--wrap_up_prompt", type=str, default="You are writing a long sequence of text. The text immediately below contains the original prompt. The text below that contains a summary of everything you've written so far. The text below that contains a copy of some of the text at the end of the text sequence you've written so far. Please write some text bringing what you've written to a conclusion.", help = "prompt to include in last API call to get it to wrap-up narrative/essay, with the original prompt, last paragraph, and a summary of all previous paragraphs as context") 
parser.add_argument("--wrap_up_without_summarization_prompt", type=str, default="You are writing a long sequence of text. The text immediately below contains the original prompt. The text below that contains a copy of some of the text at the end of the text sequence you've written so far. Please write some text bringing what you've written to a conclusion.", help = "prompt to include in last API call to get it to wrap-up narrative/essay, with only the original prompt and last paragraph as context") 


# SHOULD MAYBE PUT SOME EXCEPTION STATEMENTS HERE PUTTING LIMITS ON CONTEXT LENGTH + OUTPUT LENGTH + SUMMARIZATION LENGTH TO ENSURE THIS IS ALWAYS LESS 
# THAN MAXIMUM CONTEXT WINDOW FOR MODEL AND OTHERWISE THROW AN ERROR

args = parser.parse_args() # get object containing values of all variables specified by user (or default values if none specified by user)

os.chdir(args.directory) # Change working directory to the directory specified by user

openai.api_key = os.getenv("OPENAI_API_KEY") # Set value of OpenAI API key by retrieving value from environment variable. 


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    global encoding
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_text(with_summarization: bool) -> str:
	"""Generates text either with or without using summarization"""
	
	# Generate first block of text
	initial_prompt = args.prompt + "\n\n" + args.additional_instruction
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", 
		messages=[
			{"role": "system", "content": "You are a helpful assistant."}, 
			{"role": "user", "content": initial_prompt}], 
		temperature=args.temperature, 
		max_tokens = args.max_output_tokens_generation
	)['choices'][0]['message']['content'] 

	# If using summarization, summarize initial block of text
	if with_summarization == True:
		summarization_prompt = args.initial_summarization_prompt.format(args.max_output_tokens_summarization) + "\n\n" + response

		summary = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			messages=[
				{"role": "system", "content": "You are a helpful assistant."}, 
				{"role": "user", "content": summarization_prompt}], 
			temperature=args.temperature, 
			max_tokens = args.max_output_tokens_summarization
		)['choices'][0]['message']['content']
	
	# Extract context from initial block of text to use as input when generating next block of text.  	
	if num_tokens_from_string(response) <= args.context_length:
		context = response
	else:
		tokenized_response = encoding.encode(response)
		context = encoding.decode(tokenized_response[-args.context_length:])

	# Start a log of prompts, responses, and summaries to use for debugging and evaluating quality of summaries as necessary
	if with_summarization == True: 
		response_log = "GENERATION PROMPT 1:" + "\n\n" + initial_prompt + "\n\n" + "RESPONSE:" + "\n\n" + response + "\n\n" + "SUMMARIZATION PROMPT:" + "\n\n" + summarization_prompt + "\n\n" + "SUMMARY OF RESPONSES SO FAR:" + "\n\n" + summary + "\n\n" 
	else: 
		response_log = "GENERATION PROMPT 1:" + "\n\n" + initial_prompt + "\n\n" + "RESPONSE:" + "\n\n" + response + "\n\n" 
		
	# Begin compiling generated text into 'response' variable
	response = "PROMPT:" + "\n\n" + args.prompt + "\n\n" + "RESPONSE:" + "\n\n" + response + "\n\n"


	for i in range(2,args.num_completions - 1): # Continue generating completions, stopping before final completion
		
		# Generate continuation of previously-generated text
		if with_summarization == True:
			continuation_prompt = args.continuation_prompt + "\n\n" + "Original prompt:" + "\n\n" + args.prompt + "\n\n" + "Summary of previous text:" + "\n\n" + summary + "\n\n" + "Last {} tokens you wrote:".format(num_tokens_from_string(context)) + "\n\n" + context 
		else:
			continuation_prompt = args.continuation_without_summarization_prompt + "\n\n" + "Original prompt:" + "\n\n" + args.prompt + "\n\n" + "Last {} tokens you wrote:".format(num_tokens_from_string(context)) + "\n\n" + context
		
		continuation = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			messages=[
				{"role": "system", "content": "You are a helpful assistant."}, 
				{"role": "user", "content": continuation_prompt}], 
			temperature=args.temperature, 
			max_tokens = args.max_output_tokens_generation
		)['choices'][0]['message']['content']

		
		# Summarize text generated so far, using initial prompt, previous summary, and most recently generated text as context
		if with_summarization == True:
			summarization_prompt = args.continuation_summarization_prompt.format(args.max_output_tokens_summarization) + "\n\n" + "Original prompt:" + "\n\n" + args.prompt + "\n\n" + "Summary of previous text except for last {} tokens:".format(num_tokens_from_string(continuation)) + "\n\n" + summary + "\n\n" + "Last {} tokens you wrote:".format(num_tokens_from_string(continuation)) + "\n\n" + continuation
			summary = openai.ChatCompletion.create(
				model="gpt-3.5-turbo", 
				messages=[
					{"role": "system", "content": "You are a helpful assistant."}, 
					{"role": "user", "content": summarization_prompt}], 
				temperature=args.temperature, 
				max_tokens = args.max_output_tokens_summarization
			)['choices'][0]['message']['content']
		
		response = response + "\n\n" + continuation # Append continuation to previously-generated text 
		
		
		# Extract context from response so far to use as input when generating next block of text.  	
		if num_tokens_from_string(response) <= args.context_length:
			context = response
		else:
			tokenized_response = encoding.encode(response)
			context = encoding.decode(tokenized_response[-args.context_length:])

		
		# Update log
		if with_summarization == True:
			response_log = response_log + "GENERATION PROMPT {}:".format(i) + "\n\n" + continuation_prompt + "\n\n" + "RESPONSE:" + "\n\n" + continuation + "\n\n" + "SUMMARIZATION PROMPT:" + "\n\n" + summarization_prompt + "\n\n" +  "SUMMARY OF RESPONSES SO FAR:" + "\n\n" + summary + "\n\n" 
		else: 
			response_log = response_log + "GENERATION PROMPT {}:".format(i) + "\n\n" + continuation_prompt + "\n\n" + "RESPONSE:" + "\n\n" + continuation + "\n\n" 


	# Generate final block of text
	if with_summarization == True:
		wrap_up_prompt = args.wrap_up_prompt + "\n\n" + "Original prompt:" + "\n\n" + args.prompt + "\n\n" + "Summary of previous text:" + "\n\n" + summary + "\n\n" + "Last {} tokens you wrote:".format(num_tokens_from_string(context)) + "\n\n" + context
	else:
		wrap_up_prompt = args.wrap_up_without_summarization_prompt + "\n\n" + "Original prompt:" + "\n\n" + args.prompt + "\n\n" + "Last {} tokens you wrote:".format(num_tokens_from_string(context)) + "\n\n" + context
	final_completion = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			messages=[
				{"role": "system", "content": "You are a helpful assistant."}, 
				{"role": "user", "content": wrap_up_prompt}], 
			temperature=args.temperature, 
			max_tokens = args.max_output_tokens_generation
		)['choices'][0]['message']['content']
	response = response + "\n\n" + final_completion # Append final block of text to previously-generated text

	# Update log
	response_log = response_log + "FINAL GENERATION PROMPT:" + "\n\n" + wrap_up_prompt + "\n\n" + "FINAL RESPONSE:" + "\n\n" + final_completion # Add last response to log

	# Save response and log to txt files
	if with_summarization == True:
		filename = args.file_name + "_with_summarization.txt"
		log_filename = args.file_name + "_with_summarization_log.txt"
	else:
		filename = args.file_name + "_without_summarization.txt"
		log_filename = args.file_name + "_without_summarization_log.txt"
	with open(filename, "w") as response_file: # save response to a txt file 
		response_file.write(response)
	with open(log_filename, "w") as response_log_file: # save log of responses to a txt file
		response_log_file.write(response_log)


# Generate responses to prompt both with and without using summarization 
generate_text(with_summarization = True)
generate_text(with_summarization = False)






