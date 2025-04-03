from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

class LLM_Tester:

    def __init__(self, hf_model, max_seq_length = 2048, load_in_4bit = True):
        # configuration parameters
        self.max_seq_length = max_seq_length
        self.dtype = None
        self.load_in_4bit = load_in_4bit

        # alpaca formatting template
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        # load model and tokenizer
        self.hugging_face_model = hf_model
        self.model = AutoPeftModelForCausalLM.from_pretrained(self.hugging_face_model, load_in_4bit=self.load_in_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hugging_face_model)

    def format_result(self, result):
        return result[0].split("### Response:\n")[-1].split("<|end_of_text|>")[0]

    def generate_output(self, user_input):
        # tokenize input data with alpaca format
        inputs = self.tokenizer(
            [
                self.alpaca_prompt.format(
                    user_input,
                    "",  # input
                    "",  # output left blank for generation
                )
            ], return_tensors="pt").to("cuda")

        # generate output using the model
        outputs = self.model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        result = self.tokenizer.batch_decode(outputs)

        return self.format_result(result)
